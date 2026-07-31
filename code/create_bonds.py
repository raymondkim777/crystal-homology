import os
import pickle
import argparse
import numpy as np
import networkx as nx

from tqdm import tqdm
import warnings
from pymatgen.io.cif import CifParser
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core.periodic_table import Element
from pymatgen.core.composition import Composition
from concurrent.futures import ProcessPoolExecutor, as_completed

from find_hch import make_structure_ordered
from utils import CRYSTAL_SYSTEMS, get_num_cpus, open_write_file


DATA_DIRECTORY = None
CIF_DIRECTORY = None
STRUCTURE_DIRECTORY = None
MULTIGRAPH_DIRECTORY = None
GRAPH_DIRECTORY = None


CRYSTALNN = None
CRYSTALNN_LARGE = None


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--abs', action='store_true', help='Only focuses separately on data/abs')
    return parser.parse_args()


def fetch_cif_filenames(system: str) -> list:
    print(f"Fetching CIF files of {system} system...")
    cif_files = []

    # os.scandir() returns an iterator of DirEntry objects
    with os.scandir(f"{CIF_DIRECTORY}/{system}") as entries:
        for entry in entries:
            if not entry.is_file():
                continue
            cif_files.append(entry.name)
    return cif_files


def get_structures_from_cif(filepath: str) -> list:
    cif_parser = CifParser(filepath)
    structures = cif_parser.parse_structures()

    for struct in structures:
        check_result = cif_parser.check(struct)
        if check_result is not None:
            print(f"CIF Error: {filepath}")
            print(f"Error Message: {check_result}")
            raise ValueError(f"Struct contained in {filepath} is invalid")
    return structures


def process_one_cif(args):
    # nx_multigraph: multidigraph  -->  CGCNN input
    # nx_graph: directed graph (not multi)  -->  PH input

    # accept one tuple for multiprocessing
    system, filename, structure = args

    if structure is None:
        structure_filename = f"{CIF_DIRECTORY}/{system}/{filename}"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            structures = get_structures_from_cif(structure_filename)
        structure = structures[0].get_reduced_structure()

    ordered_structure = structure
    if not structure.is_ordered:
        ordered_structure = make_structure_ordered(structure)
    
    # ! need to adjust search_cutoff if necessary
    try:
        bonded_structure = CRYSTALNN.get_bonded_structure(
            ordered_structure, 
        )
    except ValueError as e:
        bonded_structure = CRYSTALNN_LARGE.get_bonded_structure(
            ordered_structure, 
        )
    nx_multigraph = bonded_structure.graph

    # set bond distances as weight for each edge
    for u, v, key, data in nx_multigraph.edges(keys=True, data=True):
        to_jimage = data['to_jimage']
        dist = structure.get_distance(u, v, jimage=to_jimage)
        nx_multigraph.edges[u, v, key]['weight'] = dist

    # change species identifier to species object, not label
    for node in nx_multigraph.nodes:
        if structure[node].is_ordered:
            nx_multigraph.nodes[node]['species'] = {structure[node].specie.number: 1.0}
        else:
            el_amt_dict = structure[node].species.get_el_amt_dict()
            total_amt = sum(el_amt_dict.values())
            if total_amt > 1.0:
                raise ValueError(f"[Graph Creation] total amount in site {node} > 1.0")
            nx_multigraph.nodes[node]['species'] = {
                Element(element).number: el_amt_dict[element]
                for element in el_amt_dict.keys()
            }
            # max_element = max(el_amt_dict, key=el_amt_dict.get)
            # max_element_num = Element(max_element).number
            # nx_multigraph.nodes[node]['species'] = max_element_num

    # collapse multigraph into graph
    nx_graph = nx.DiGraph(nx_multigraph)

    # remove self connections
    for node in nx_graph.nodes:
        if nx_graph.has_edge(*(node, node)):
            nx_graph.remove_edge(*(node, node))

    # save edge multiplicites --> DEPR, CGCNN can handle multiple edges
    edge_mult_dict = dict()
    for edge in nx_multigraph.edges:  # tuple
        edge_collapsed = (edge[0], edge[1])  # three entries, third one is unique key
        edge_mult_dict[edge_collapsed] = len(nx_multigraph[edge[0]][edge[1]])

    # store edge multiplicities as edge attribute
    nx.set_edge_attributes(nx_graph, values=edge_mult_dict, name='multiplicity')

    # re-implement edge weights as min. distances of all relevant bonds
    distance_matrix = structure.distance_matrix
    distances_dict = dict()
    for edge in nx_graph.edges:
        distances_dict[edge] = distance_matrix[edge[0], edge[1]]
    nx.set_edge_attributes(nx_graph, values=distances_dict, name='weight')
    
    # remove to_jimage attribute
    for u, v, data in nx_graph.edges(data=True):
        data.pop("to_jimage", None)

    # nx_graph = nx_graph.to_undirected()
    crystal_id = filename[:-4]

    return crystal_id, structure, nx_multigraph, nx_graph


def init_crystalnn(crystalnn, crystalnn_large):
    global CRYSTALNN, CRYSTALNN_LARGE
    CRYSTALNN = crystalnn
    CRYSTALNN_LARGE = crystalnn_large


def construct_crystalnn_graph() -> None:
    workers = get_num_cpus()
    print(f"Using {workers} worker processes")

    # parameters selected for structural bonds, not chemical bonds
    crystalnn = CrystalNN(
        distance_cutoffs=None, 
        x_diff_weight=0, 
        porous_adjustment=False,
        search_cutoff=11,  # default 7, but ERROR: No Voronoi neighbors found for site
    )

    crystalnn_large = CrystalNN(
        distance_cutoffs=None, 
        x_diff_weight=0, 
        porous_adjustment=False,
        search_cutoff=16,  # default 7, but ERROR: No Voronoi neighbors found for site
    )

    unordered_id_list = []
    for system in CRYSTAL_SYSTEMS:

        system_structures = dict()
        system_multigraphs = dict()
        system_graphs = dict()

        cif_files = fetch_cif_filenames(system)

        struct_exist = False
        if os.path.isfile(f"{STRUCTURE_DIRECTORY}/{system}.pkl"):
            print(f"Structure directory exists!")
            struct_exist = True
            with open(f'{STRUCTURE_DIRECTORY}/{system}.pkl', 'rb') as file:
                system_structs = pickle.load(file)
            tasks = [(system, filename, system_structs[filename[:-4]]) for filename in cif_files]
        else:
            print(f"Structure directory doesn't exist, need to parse CIFs")
            tasks = [(system, filename, None) for filename in cif_files]
                    
        print(f"Creating {'structs and ' if not struct_exist else ''}graphs of {system} system...")
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=init_crystalnn, 
            initargs=(crystalnn, crystalnn_large),
        ) as executor:
            futures = [
                executor.submit(process_one_cif, task)
                for task in tasks
            ]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f'{system}: ',
            ):
                crystal_id, structure, nx_multigraph, nx_graph = future.result()
            # results = executor.map(process_one_cif, tasks, chunksize=8)
            # for crystal_id, structure, nx_multigraph, nx_graph in tqdm(results, total=len(tasks)):
                system_structures[crystal_id] = structure
                system_multigraphs[crystal_id] = nx_multigraph
                system_graphs[crystal_id] = nx_graph
                if not structure.is_ordered:
                    unordered_id_list.append(crystal_id)
        
        # save structures/graphs as pickles
        if not struct_exist:
            structure_filepath = open_write_file(STRUCTURE_DIRECTORY, f'{system}.pkl')
            with open(structure_filepath, 'wb') as f:
                pickle.dump(system_structures, f)

        multigraph_filepath = open_write_file(MULTIGRAPH_DIRECTORY, f'{system}.pkl')
        with open(multigraph_filepath, 'wb') as f:
            pickle.dump(system_multigraphs, f)
        
        graph_filepath = open_write_file(GRAPH_DIRECTORY, f'{system}.pkl')
        with open(graph_filepath, 'wb') as f:
            pickle.dump(system_graphs, f)
        
    print(f"Unordered structures: {len(unordered_id_list)}")
    with open(f'{DATA_DIRECTORY}/unordered.txt', 'w') as f:
        f.write("MP/COD IDs of unordered parsed CIF structures:\n")
        for cod_id in unordered_id_list:
            f.write(f"{cod_id}\n")
        f.write("END_LIST")


def check_structures() -> None:
    for system in CRYSTAL_SYSTEMS:
        struct_mult_list = []
        cif_files = fetch_cif_filenames(system)
        for filename in tqdm(cif_files):
            structure_filename = f"{CIF_DIRECTORY}/{system}/{filename}"
            structures = get_structures_from_cif(structure_filename)
            if len(structures) > 1:
                struct_mult_list.append(filename[:-4])
        filepath = open_write_file(f'data/struct-issues', f"{system}")
        with open(filepath, 'w') as f:
            f.write("\n".join(struct_mult_list))


if __name__ == "__main__":
    args = _parse_args()

    DATA_DIRECTORY = "data/pretrain"
    if args.abs:
        DATA_DIRECTORY = "data/abs"

    CIF_DIRECTORY = f"{DATA_DIRECTORY}/cif"
    STRUCTURE_DIRECTORY = f"{DATA_DIRECTORY}/structs"
    MULTIGRAPH_DIRECTORY = f"{DATA_DIRECTORY}/graphs-multi"
    GRAPH_DIRECTORY = f"{DATA_DIRECTORY}/graphs"

    label = 'ABS' if args.abs else 'PRETRAIN'
    print(f"----------- {label} -----------")
    construct_crystalnn_graph()
    print(f"----------- {label} END -----------")
    # check_structures()