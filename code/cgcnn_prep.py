import argparse
import random
import pickle
import csv
import json
import numpy as np
import networkx as nx
import warnings
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed
from create_bonds import get_structures_from_cif
from vectorizers import IM_BANDWIDTH, IM_RESOLUTION, fit_image_transformers
from utils import CRYSTAL_SYSTEMS, PREDICT, DIMENSION_CNT, get_num_cpus, open_write_file


CGCNN_DATAPATH = 'cgcnn/data/graph_data'
# CGCNN_DATAPATH = 'cgcnn/data/example_graph_data'


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', action='store_true', help='Saves graph data to CGCNN data path')
    parser.add_argument('--vector', action='store_true', help='Saves vectorizations to CGCNN data path')
    parser.add_argument('--bound', action='store_true', help='Saves persistence bounds to CGCNN data path')
    return parser.parse_args()


def unpack_system_graphs(args):
    system = args
    graph_system_dict = dict()
    
    with open(f'data/graphs-multi/{system}.pkl', 'rb') as file:
        graph_system_dict = pickle.load(file)

    for mp_id, graph in graph_system_dict.items():
        structure_filename = f'data/cif/{system}/{mp_id}.cif'

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            structure = get_structures_from_cif(structure_filename)[0]

        graph_system_dict[mp_id] = {
            'graph': graph.to_undirected(),     # UNDIRECTED (for message passing)
            'system': system, 
            'lattice_matrix': structure.lattice.matrix,
        }
    
    return system, graph_system_dict


def unpack_all_graphs() -> dict:
    num_workers = get_num_cpus()
    graph_dict = {}

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(unpack_system_graphs, system): system
            for system in CRYSTAL_SYSTEMS
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc='Unpacking systems'):
            system, graph_system_dict = future.result()
            graph_dict.update(graph_system_dict)
            print(f"\nFinished unpacking {system} system with {len(graph_system_dict)} graphs")
        # results = executor.map(unpack_system_graphs, CRYSTAL_SYSTEMS)

        # for system, graph_system_dict in tqdm(results, total=len(CRYSTAL_SYSTEMS)):
        #     graph_dict.update(graph_system_dict)

    return graph_dict


def collect_all_diagrams():
    '''Collects all persistence diagrams from .pkl files from all systems into one dict/list.'''
    all_diagrams_dict = dict()
    all_diagrams_list = []

    for system in CRYSTAL_SYSTEMS:
        with open(f'data/diagrams/{system}.pkl', 'rb') as file:
            diagrams = pickle.load(file)
        all_diagrams_dict = all_diagrams_dict | diagrams
        all_diagrams_list += list(diagrams.values())

    return all_diagrams_dict, all_diagrams_list,


def remove_diagram_padding(diagram, eps=1e-12):
    '''Removes all infinite values & (b, d) such that b == d'''
    diagram = np.asarray(diagram, dtype=float)

    finite_mask = np.isfinite(diagram).all(axis=1)
    persistence_mask = (diagram[:, 1] - diagram[:, 0]) > eps
    final_diagram = diagram[finite_mask & persistence_mask]
    # NOTE: final diagram may be empty
    return final_diagram


def process_all_diagrams(diagram_dict: dict) -> list:
    '''
    Removes all diagram giotto-tda padding (b == d), organizes triplets into
    separate dimensions, and removes dimension field. Applied to list of diagrams.
    Input: [diagram1, diagram2, ...] where diagram = [[b, d, dim], ...]
    Output: [{id1: h0_diagram1, id2: h0_diagram2, ...}, {id1: h1_diagram1, ...}, ...] 
            where each Hn diagram is [[b, d], ...]
    '''
    dimension_array = []
    for dim in range(DIMENSION_CNT):
        dim_diagrams = dict()
        for mp_id, diagram in diagram_dict.items():
            triplets_in_dim = diagram[diagram[:, 2] == dim]
            doubles_in_dim = triplets_in_dim[:, :2]
            final_diagram = remove_diagram_padding(doubles_in_dim)
            dim_diagrams[mp_id] = final_diagram
        dimension_array.append(dim_diagrams)
    return dimension_array


def homogenize_diags_in_dict(diagram_dict: dict) -> dict:
    '''Homogenizes np shape for given diagram dict (one dimension)'''
    num_diagrams = len(diagram_dict.keys())
    lengths = np.array([diag.shape[0] for diag in diagram_dict.values()])
    max_n = lengths.max()

    padded = np.full(
        shape=(num_diagrams, max_n, 2), 
        fill_value = 0, 
        dtype=np.float32
    )
    final_dict = dict()
    for i, (id, diag) in enumerate(diagram_dict.items()):
        n = diag.shape[0]
        padded[i, : n, :] = diag
        final_dict[id] = padded[i]
    return final_dict


def reorganize_list_of_dicts(list_of_dicts: list) -> dict:
    '''
    Input: [{id1: h0_diagram1, id2: h0_diagram2, ...}, {id1: h1_diagram1, ...}, ...] 
            where each Hn diagram is [[b, d], ...]
    Output: {id1: [h0_diagram1, h1_diagram1, h2_diagram1], ...}
            Twhere each Hn diagram is [[b, d], ...]
    '''
    diagram_dict = dict()
    for dim in range(DIMENSION_CNT):
        for mp_id, diagram in list_of_dicts[dim].items():
            if mp_id not in diagram_dict.keys():
                diagram_dict[mp_id] = []
            diagram_dict[mp_id].append(diagram)
    return diagram_dict


def retrieve_diagrams():
    diagrams_dict, _ = collect_all_diagrams()
    processed_diagrams_by_dim = process_all_diagrams(diagrams_dict)     # [mp_id][dim] --> processed diagram
    padded_diagram_dict_by_dim = [homogenize_diags_in_dict(processed_diagrams_by_dim[dim]) for dim in range(DIMENSION_CNT)]
    return reorganize_list_of_dicts(padded_diagram_dict_by_dim)


def write_id_prop_mask():
    doc_dict = dict()
    for system in CRYSTAL_SYSTEMS:
        print(f"Unpacking {system} system docs...")
        with open(f'data/mp-subset/{system}.pkl', 'rb') as file:
            system_dict = pickle.load(file)
        doc_dict.update(system_dict)
    
    csv_prop_data = []
    csv_mask_data = []
    system_to_int = {CRYSTAL_SYSTEMS[idx]: idx for idx in range(len(CRYSTAL_SYSTEMS))}

    print(f"Computing id_prop.csv and id_mask.csv...")
    for mp_id, doc in tqdm(doc_dict.items()):
        prop_dict = {
            'system': system_to_int[str(doc['symmetry'].crystal_system).lower()],
            'bm_voigt': doc['bulk_modulus']['voigt'] if doc['bulk_modulus'] is not None else 0.0,
            'bm_reuss': doc['bulk_modulus']['reuss'] if doc['bulk_modulus'] is not None else 0.0,
            'bm_vrh': doc['bulk_modulus']['vrh'] if doc['bulk_modulus'] is not None else 0.0,
            'direct_gap': doc['bandstructure'].latimer_munro.direct_gap 
            if doc['bandstructure'] is not None and doc['bandstructure'].latimer_munro is not None 
            else 0.0,
            'band_gap': doc['band_gap'] if doc['band_gap'] is not None else 0.0,
            'efermi': doc['efermi'] if doc['efermi'] is not None else 0.0,
            'is_gap_direct': 1 if doc['is_gap_direct'] is not None and doc['is_gap_direct'] else 0,
        }
        mask_dict = {
            'system': 1,
            'bm_voigt': int(doc['bulk_modulus'] is not None),
            'bm_reuss': int(doc['bulk_modulus'] is not None),
            'bm_vrh': int(doc['bulk_modulus'] is not None),
            'direct_gap': int(doc['bandstructure'] is not None and doc['bandstructure'].latimer_munro is not None),
            'band_gap': int(doc['band_gap'] is not None),
            'efermi': int(doc['efermi'] is not None),
            'is_gap_direct': int(doc['is_gap_direct'] is not None),
        }
        csv_prop_row = [mp_id[3:]]
        csv_mask_row = [mp_id[3:]]

        # ensure order is same as PREDICT in utils.py
        for prop in PREDICT:
            csv_prop_row.append(prop_dict[prop])
            csv_mask_row.append(mask_dict[prop])
        csv_prop_data.append(csv_prop_row)
        csv_mask_data.append(csv_mask_row)
    
    print(f"Writing id_prop.csv and id_mask.csv...")
    csv_prop_filepath = open_write_file(CGCNN_DATAPATH, 'id_prop.csv')
    with open(csv_prop_filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(csv_prop_data)

    csv_mask_filepath = open_write_file(CGCNN_DATAPATH, 'id_mask.csv')
    with open(csv_mask_filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(csv_mask_data)

    # save PREDICT list to CGCNN data path
    predict_filepath = open_write_file(f"{CGCNN_DATAPATH}", f'predict.pkl')
    with open(predict_filepath, 'wb') as f:
        pickle.dump(PREDICT, f)


def graph_process(vector=False):
    """
    Saves all graphs with labeled crystal systems in CGCNN data folder. 
    Computes and prints required bounds for GraphData.
    Graph structure:
    mp_id: {
        graph: <graph>, 
        system: <system>,
        lattice_matrix: <structure.lattice.matrix>,
    }
    Optionally saves diagrams and vectorizations as pickle files in CGCNN data folder. 
    """
    graph_dict = unpack_all_graphs()
    print(f"Saving files to {CGCNN_DATAPATH}")
    # save graphs to CGCNN data folder
    for mp_id, value in graph_dict.items():
        cgcnn_datapath = open_write_file(f"{CGCNN_DATAPATH}/graphs", f'{mp_id}.pkl')
        with open(cgcnn_datapath, 'wb') as f:
            pickle.dump(value, f)
    
    # multitask regression/classification id_prop and id_mask
    write_id_prop_mask()

    if vector:
        diagram_dict = retrieve_diagrams()
        image_dict, landscape_dict = dict(), dict()
        for system in CRYSTAL_SYSTEMS:
            with open(f'data/images/{system}.pkl', 'rb') as file:
                images = pickle.load(file)
            image_dict = image_dict | images  # [mat_id][dim]
            
            with open(f'data/landscapes/{system}.pkl', 'rb') as file:
                landscapes = pickle.load(file)
            landscape_dict = landscape_dict | landscapes  # [mat_id][dim]

        cgcnn_diagram_datapath = open_write_file(f"{CGCNN_DATAPATH}", f'diagrams.pkl')
        with open(cgcnn_diagram_datapath, 'wb') as f:
            pickle.dump(diagram_dict, f)

        cgcnn_image_datapath = open_write_file(f"{CGCNN_DATAPATH}", f'images.pkl')
        with open(cgcnn_image_datapath, 'wb') as f:
            pickle.dump(image_dict, f)

        cgcnn_landscape_datapath = open_write_file(f"{CGCNN_DATAPATH}", f'landscapes.pkl')
        with open(cgcnn_landscape_datapath, 'wb') as f:
            pickle.dump(landscape_dict, f)


def pad_bounds(bound_x, bound_y, eps=0.001):
    if bound_x == bound_y:
        bound_x -= eps
        bound_y -= eps
    return (bound_x, bound_y)


def save_bounds():
    image_transformers = fit_image_transformers(IM_BANDWIDTH, IM_RESOLUTION)
    # self.image_bnds = [
    #     ((-1.0e-03, 1.0e-03), (0.54836184, 16.7195282)), 
    #     ((1.22288406e+00, 1.44270658e+01), (2.38418579e-07, 1.54966436e+01)), 
    #     ((1.71408939e+00, 1.52640104e+01), (2.38418579e-07, 1.49896426e+01))
    # ]
    image_bnds_list = []
    for dim in range(DIMENSION_CNT):
        bounds = image_transformers[dim].im_range_fixed_
        bounds_tuple = (pad_bounds(bounds[0], bounds[1]), pad_bounds(bounds[2], bounds[3]))
        image_bnds_list.append(bounds_tuple)
    
    # save JSON
    file_path = open_write_file(CGCNN_DATAPATH, 'bounds.pkl')
    with open(file_path, "wb") as f:
        pickle.dump(image_bnds_list, f)
            

if __name__ == "__main__":
    args = _parse_args()

    if args.save:
        graph_process(args.vector)
    if args.bound:
        save_bounds()