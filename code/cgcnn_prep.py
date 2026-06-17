from tqdm import tqdm
import random
import argparse
import numpy as np
import networkx as nx
from tqdm import tqdm
import pickle
from utils import CRYSTAL_SYSTEMS, DIMENSION_CNT, open_write_file, plot_nxgraph


CGCNN_DATAPATH = 'cgcnn/data/graph_data'


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', action='store_true', help='Saves graph data to CGCNN data path')
    parser.add_argument('--vector', action='store_true', help='Saves vectorizations to CGCNN data path')
    return parser.parse_args()


def unpack_all_graphs() -> dict:
    graph_dict = dict()
    for system in CRYSTAL_SYSTEMS:
        print(f"Unpacking {system} system...")
        with open(f'data/graphs-multi/{system}.pkl', 'rb') as file:
            graph_dict[system] = pickle.load(file)
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


# def process_all_diagrams(diagram_dict: dict):
#     '''
#     Removes all diagram giotto-tda padding (b == d), organizes triplets into
#     separate dimensions, and removes dimension field. Applied to list of diagrams.
#     Input: {id1: diagram1, id2: diagram2, ...} where diagram = [[b, d, dim], ...]
#     Output: {id1: [h0_diagram1, h1_diagram1, h2_diagram1], ...}
#             where each Hn diagram is [[b, d], ...]
#     '''
#     dim_diagrams = dict()
#     for mp_id, diagram in diagram_dict.items():
#         dim_diagrams[mp_id] = []
#         for dim in range(DIMENSION_CNT):
#             triplets_in_dim = diagram[diagram[:, 2] == dim]
#             doubles_in_dim = triplets_in_dim[:, :2]
#             final_diagram = remove_diagram_padding(doubles_in_dim)
#             dim_diagrams[mp_id].append(final_diagram)
#     return dim_diagrams


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
#             where each Hn diagram is [[b, d], ...]
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


def graph_process(save=False, vector=False) -> dict:
    """
    Saves all graphs with labeled crystal systems in CGCNN data folder. 
    Computes and prints required bounds for GraphData.
    Graph structure:
    mp_id: {
        graph: <graph>, 
        system: <system>,
        ... (additional properties to be added)
    }
    Optionally saves diagrams and vectorizations as pickle files in CGCNN data folder. 
    """
    graph_dict = unpack_all_graphs()
    new_graph_dict = dict()  # mp_id: {graph: <graph>, system: 'system'}

    # UNDIRECTED edges
    for system in CRYSTAL_SYSTEMS:
        for key, graph in graph_dict[system].items():
            new_graph_dict[key] = {
                'graph': graph.to_undirected(),     # UNDIRECTED (for message passing)
                'system': system
            }
    
    # computing bounds
    max_num_nbr = 0
    max_bond_dist = 0

    print("Computing bounds...")
    for key, value in tqdm(new_graph_dict.items()):
        graph = value['graph']
        # max_degree = max(d for _, d in graph.out_degree())  # for digraphss
        max_degree = max(d for _, d in graph.degree())
        max_num_nbr = max(max_num_nbr, max_degree)

        weights = [0] + [data['weight'] for _, _, data in graph.edges(data=True)]
        if len(weights) == 0:
            # print(key)
            pass
        max_dist = max(weights)
        max_bond_dist = max(max_bond_dist, max_dist)

    print("Maximum Neighbor Cnt:", max_num_nbr)
    print("Maximum Bond Distance:", max_bond_dist)

    if save:
        print(f"Saving files to {CGCNN_DATAPATH}")
        # save graphs to CGCNN data folder
        for mp_id, value in new_graph_dict.items():
            cgcnn_datapath = open_write_file(f"{CGCNN_DATAPATH}/graphs", f'{mp_id}.pkl')
            with open(cgcnn_datapath, 'wb') as f:
                pickle.dump(value, f)
        
        # create id_prop.csv
        system_to_int = {CRYSTAL_SYSTEMS[idx]: idx for idx in range(len(CRYSTAL_SYSTEMS))}
        
        csv_filepath = open_write_file(CGCNN_DATAPATH, 'id_prop.csv')
        with open(csv_filepath, 'w') as f:
            # CRYSTAL SYSTEM - classification
            for mp_id, value in new_graph_dict.items():
                f.write(f"{mp_id[3:]}, {system_to_int[value['system']]}\n")

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
            
            



def bid_test():
    '''
    Testing all digraphs to check for bidirectional edges.
    RESULT: No bidirectional edges in all 7000 digraphs.
    '''
    graph_dict = unpack_all_graphs()
    new_graph_dict = dict()  # mp_id: {graph: <graph>, system: 'system'}
    for system in CRYSTAL_SYSTEMS:
        for key, value in graph_dict[system].items():
            new_graph_dict[key] = {
                'graph': value, 
                'system': system
            }
    
    no_bidirectional = True
    for key, value in new_graph_dict.items():
        graph = value['graph']
        has_bidirectional = any(graph.has_edge(v, u) for u, v in graph.edges() if u != v)
        if has_bidirectional:
            no_bidirectional = False
            print(key)
    print(no_bidirectional)


if __name__ == "__main__":
    args = _parse_args()
    graph_process(args.save, args.vector)
    # bid_test()