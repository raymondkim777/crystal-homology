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
from concurrent.futures import ProcessPoolExecutor

from utils import CRYSTAL_SYSTEMS, get_num_cpus, open_write_file


# DATA_DIRECTORY = "data/pretrain"
# DATA_DIRECTORY = "data/abs"
# CIF_DIRECTORY = f"{DATA_DIRECTORY}/cif"
# MULTIGRAPH_DIRECTORY = f"{DATA_DIRECTORY}/graphs-multi"
# GRAPH_DIRECTORY = f"{DATA_DIRECTORY}/graphs"

DATA_DIRECTORY = None
CIF_DIRECTORY = None
MULTIGRAPH_DIRECTORY = None
GRAPH_DIRECTORY = None


CRYSTALNN = None


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--abs', action='store_true', help='Only focuses separately on data/abs')
    parser.add_argument('--plqy', action='store_true', help='Only focuses separately on data/plqys')
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


def get_structures_from_cif_plqy(filepath: str) -> list:
    # ! parse_structures() returns "Incorrect stoichiometry" error
    # ! --> bypass occupancy checks
    cif_parser = CifParser(filepath, occupancy_tolerance=np.inf)
    structures = cif_parser.parse_structures(check_occu=False)
    if len(structures) > 1:
        print(f"[PLQY Structures] File {filepath} generates multiple structures")
    
    for site in structures[0]:
        total_occ = sum(site.species.values())
        if total_occ > 1.0:
            new_species_dict = {sp: occ / total_occ for sp, occ in site.species.items()}
            new_species = Composition.from_weight_dict(new_species_dict)
            site.species = new_species
    return structures


def process_one_cif(args):
    # accept one tuple for multiprocessing
    system, filename, plqy = args

    structure_filename = f"{CIF_DIRECTORY}/{system}/{filename}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if plqy:
            structures = get_structures_from_cif_plqy(structure_filename)
        else:
            structures = get_structures_from_cif(structure_filename)
    
    # transform into multi-digraph
    # structure = structures[0]
    structure = structures[0].get_reduced_structure()
    bonded_structure = CRYSTALNN.get_bonded_structure(
        structure,
        on_disorder='take_max_species', 
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
            nx_multigraph.nodes[node]['species'] = structure[node].specie.number
        else:
            el_amt_dict = structure[node].species.get_el_amt_dict()
            max_element = max(el_amt_dict, key=el_amt_dict.get)
            max_element_num = Element(max_element).number
            nx_multigraph.nodes[node]['species'] = max_element_num

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
    
    crystal_id = filename[:-4]

    return crystal_id, nx_multigraph, nx_graph


def init_crystalnn(crystalnn):
    global CRYSTALNN
    CRYSTALNN = crystalnn


def construct_crystalnn_graph(plqy=False) -> None:
    workers = get_num_cpus()
    print(f"Using {workers} worker processes")

    # parameters selected for structural bonds, not chemical bonds
    crystalnn = CrystalNN(
        distance_cutoffs=None, 
        x_diff_weight=0, 
        porous_adjustment=False,
        search_cutoff=11,  # default 7, but ERROR: No Voronoi neighbors found for site
    )

    for system in CRYSTAL_SYSTEMS:

        system_multigraphs = dict()
        system_graphs = dict()

        cif_files = fetch_cif_filenames(system)
        tasks = [(system, filename, plqy) for filename in cif_files]
                    
        print(f"Creating graphs of {system} system...")
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=init_crystalnn, 
            initargs=(crystalnn,),
        ) as executor:
            results = executor.map(process_one_cif, tasks, chunksize=8)

            for crystal_id, nx_multigraph, nx_graph in tqdm(results, total=len(tasks)):
                system_multigraphs[crystal_id] = nx_multigraph
                system_graphs[crystal_id] = nx_graph
        
        # save graphs as pickles
        multigraph_filepath = open_write_file(MULTIGRAPH_DIRECTORY, f'{system}.pkl')
        with open(multigraph_filepath, 'wb') as f:
            pickle.dump(system_multigraphs, f)
        
        graph_filepath = open_write_file(GRAPH_DIRECTORY, f'{system}.pkl')
        with open(graph_filepath, 'wb') as f:
            pickle.dump(system_graphs, f)


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
    assert not args.abs or not args.plqy, "Can only choose one of abs/plqy"

    DATA_DIRECTORY = "data/pretrain"
    if args.abs:
        DATA_DIRECTORY = "data/abs"
    if args.plqy:
        DATA_DIRECTORY = "data/plqy"

    CIF_DIRECTORY = f"{DATA_DIRECTORY}/cif"
    MULTIGRAPH_DIRECTORY = f"{DATA_DIRECTORY}/graphs-multi"
    GRAPH_DIRECTORY = f"{DATA_DIRECTORY}/graphs"

    construct_crystalnn_graph(plqy=args.plqy)
    # check_structures()