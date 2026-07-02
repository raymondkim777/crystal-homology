import os
import json
import argparse
import numpy as np
import networkx as nx
from tqdm import tqdm
import pickle
import warnings

from gudhi.sklearn import RipsPersistence
from gudhi.representations import DiagramSelector
from gtda.homology import FlagserPersistence
from gtda.plotting import plot_diagram

from pymatgen.io.cif import CifParser
from pymatgen.core.periodic_table import Element
from pymatgen.core.composition import Composition

from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import CRYSTAL_SYSTEMS, DIMENSION_CNT, get_num_cpus, open_write_file, get_max_dist


# DATA_DIRECTORY = "data/pretrain"
# DATA_DIRECTORY = "data/abs"
# GRAPH_DIRECTORY = f"{DATA_DIRECTORY}/graphs"
# DIAGRAM_DIRECTORY = f"{DATA_DIRECTORY}/diagrams"
# MAX_DIST = get_max_dist(DATA_DIRECTORY)

DATA_DIRECTORY = None
CIF_DIRECTORY = None
GRAPH_DIRECTORY = None
STRUCTURE_DIRECTORY = None
DIAGRAM_GRAPH_DIRECTORY = None
DIAGRAM_POINT_DIRECTORY = None
MAX_DIST = None


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='graph', help="how to create complex [graph, point]")
    parser.add_argument('--abs', action='store_true', help='constructs diagrams for absorption data')
    parser.add_argument('--plqy', action='store_true', help='constructs diagrams for plqy data')
    parser.add_argument('--plqy-full', action='store_true', help='constructs diagrams for plqy-full data')
    return parser.parse_args()


############### DIRECTED GRAPH PERSISTENCE ###############


def unpack_all_graphs() -> dict:
    graph_dict = dict()
    for system in CRYSTAL_SYSTEMS:
        with open(f'{GRAPH_DIRECTORY}/{system}.pkl', 'rb') as file:
            graph_dict[system] = pickle.load(file)
    return graph_dict


def convert_graph_to_adj_mat(graph: nx.DiGraph) -> np.ndarray:
    
    adj_mat = nx.to_numpy_array(graph, weight='weight', nonedge=np.inf)
    # set diagonals to 0 (just in case)
    np.fill_diagonal(adj_mat, 0)
    return adj_mat


def plot_persistence_diagram(mp_id, diagram) -> None:
    fig = plot_diagram(diagram)
    # fig.show()
    fig.save(f"{DIAGRAM_GRAPH_DIRECTORY}/diagram_{mp_id}.png")


def compute_persistence_diagrams_graph() -> None:
    print(f"------ GRAPH PERSISTENCE ------")
    n_jobs = get_num_cpus()
    print(f"Using {n_jobs} job processes")
    
    print(f"Unpacking all graphs...")
    graph_dict = unpack_all_graphs()

    flagser = FlagserPersistence(
        homology_dimensions=tuple(range(DIMENSION_CNT)),
        directed=True,
        filtration='max', 
        coeff=2, 
        max_edge_weight=MAX_DIST,
        infinity_values=None, 
        n_jobs=n_jobs   # parallel processing
    )

    # fit flagser to all graphs
    adj_mat_dict = dict()  # all graphs
    adj_mat_system = dict()  # all graphs divided by system
    for system in CRYSTAL_SYSTEMS:
        adj_mat_system[system] = dict()
        for mat_id, graph in graph_dict[system].items():
            adj_mat = convert_graph_to_adj_mat(graph) 
            adj_mat_system[system][mat_id] = adj_mat
            adj_mat_dict[mat_id] = adj_mat

    adj_mat_list = list(adj_mat_dict.values())
    flagser.fit(adj_mat_list)

    # compute diagrams for each system
    for system in CRYSTAL_SYSTEMS:
        print(f"Computing PD for {system}...")
        
        adj_mat_list = list(adj_mat_system[system].values())
        if len(adj_mat_list) == 0:
            print(f"No graphs for {system} system!")
            diagrams_with_id = dict()
            diag_filepath = open_write_file(DIAGRAM_GRAPH_DIRECTORY, f'{system}.pkl')
            with open(diag_filepath, 'wb') as f:
                pickle.dump(diagrams_with_id, f)
            continue

        diagrams = flagser.transform(adj_mat_list)

        # match diagrams to mat_id
        keys = list(adj_mat_system[system].keys())
        diagrams_with_id = {
            keys[i] : diagrams[i]
            for i in range(len(diagrams))
        }

        # print diagram info
        print(f"diagram cnt: {len(diagrams)}")
        graph_diagram = diagrams[0]
        print("Persistence Diagram Shape:", graph_diagram.shape)
        print("Points (Birth, Death, Homology Dimension):\n", graph_diagram)
    
        # save diagrams dict as pickle
        diag_filepath = open_write_file(DIAGRAM_GRAPH_DIRECTORY, f'{system}.pkl')
        with open(diag_filepath, 'wb') as f:
            pickle.dump(diagrams_with_id, f)
    print(f"------ END PERSISTENCE ------")


############### POINT CLOUD PERSISTENCE ###############
    

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


def get_structures_from_cif_plqy_full(filepath: str) -> list:
    cif_parser = CifParser(filepath, occupancy_tolerance=1.1)
    structures = cif_parser.parse_structures()
    
    if len(structures) > 1:
        print(f"[PLQY Structures] File {filepath} generates multiple structures")
    return structures


def compute_dist_mat_for_cif(args):
    # accept one tuple for multiprocessing
    system, filename, structure, plqy, plqy_full = args

    if structure is None:
        print(f"extracting structure for {filename}")
        # extract structure from CIF
        structure_filename = f"{CIF_DIRECTORY}/{system}/{filename}"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if plqy:
                structures = get_structures_from_cif_plqy(structure_filename)
            elif plqy_full:
                structures = get_structures_from_cif_plqy_full(structure_filename)
            else:
                structures = get_structures_from_cif(structure_filename)
        structure = structures[0].get_reduced_structure()

    # get distance matrix for each pair of frac. coordinates
    n = len(structure)
    dist_mat = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            dist, jimage = structure.lattice.get_distance_and_image(
                structure.frac_coords[i], 
                structure.frac_coords[j],
            )
            dist_mat[i, j] = dist
            dist_mat[j, i] = dist

    return filename[:-4], dist_mat


def cif_to_dist_mat(plqy=False, plqy_full=False):
    n_workers = get_num_cpus()
    dist_dict = dict()

    print(f"Unpacking system structures into cartesian coordinates...")
    for system in CRYSTAL_SYSTEMS:
        struct_exist = False
        if os.path.isfile(f"{STRUCTURE_DIRECTORY}/{system}.pkl"):
            print(f"Struct directory exists!")
            struct_exist = True
            with open(f'{STRUCTURE_DIRECTORY}/{system}.pkl', 'rb') as file:
                system_structs = pickle.load(file)
        else:
            print(f"Structure directory doesn't exist, need to parse CIFs")
        
        system_dist_mats = dict()

        cif_files = fetch_cif_filenames(system)
        if struct_exist:
            tasks = [(system, filename, system_structs[filename[:-4]], plqy, plqy_full) for filename in cif_files]
        else:
            tasks = [(system, filename, None, plqy, plqy_full) for filename in cif_files]

        # with ProcessPoolExecutor(
        #     max_workers=n_workers,
        # ) as executor:
        #     futures = [
        #         executor.submit(compute_dist_mat_for_cif, task)
        #         for task in tasks
        #     ]
        #     for future in tqdm(
        #         as_completed(futures),
        #         total=len(futures),
        #         desc=f"PC for {system}: ",
        #     ):
        #         crystal_id, dist_mat = future.result()
        #         system_dist_mats[crystal_id] = dist_mat

        with ProcessPoolExecutor(
            max_workers=n_workers, 
        ) as executor:
            results = executor.map(compute_dist_mat_for_cif, tasks)

            for crystal_id, dist_mat in tqdm(results, total=len(tasks), desc=f"PC for {system}: "):
                system_dist_mats[crystal_id] = dist_mat
        
        dist_dict[system] = system_dist_mats
    return dist_dict


def compute_persistence_diagrams_point(plqy=False, plqy_full=False):
    print(f"------ POINT CLOUD PERSISTENCE ------")
    n_workers = get_num_cpus()
    print(f"Using {n_workers} job processes")

    # compute distance matrices for each crystal for each system
    dist_dict = cif_to_dist_mat(plqy=plqy, plqy_full=plqy_full)

    rips = RipsPersistence(
        homology_dimensions=tuple(range(DIMENSION_CNT)),
        threshold=MAX_DIST, 
        input_type='full distance matrix', 
        n_jobs=n_workers,
    )

    # compute diagrams for each system
    for system in CRYSTAL_SYSTEMS:
        print(f"Computing PD for {system}")

        dist_mat_system = list(dist_dict[system].values())
        if len(dist_mat_system) == 0:
            print(f"No crystals for {system} system!")
            diagrams_with_id = dict()
            diag_filepath = open_write_file(DIAGRAM_POINT_DIRECTORY, f'{system}.pkl')
            with open(diag_filepath, 'wb') as f:
                pickle.dump(diagrams_with_id, f)
            continue
        
        # fit & transform rips to each crystal (point cloud)
        diagrams = rips.fit_transform(dist_mat_system)
        # [crystal1, crystal2, ...] where crystaln = [h0_diag, h1_diag, h2_diag]

        diagrams_filtered = []
        diag_select = DiagramSelector(
            use=True, 
            point_type='finite', 
            limit=MAX_DIST,     # + 1??
        )
        for diag_crystal in diagrams:
            diagrams_filtered.append(diag_select.fit_transform(diag_crystal))

        # match diagrams to crystal id (format with multiple dimensions)
        keys = list(dist_dict[system].keys())
        diagrams_with_id = {
            keys[i]: diagrams_filtered[i]
            for i in range(len(diagrams_filtered))
        }

        # print diagram info
        print(f"diagram cnt: {len(diagrams)}")
        point_diagram_list = diagrams[0]
        print("Persistence Diagram Shape for H0:", point_diagram_list[0].shape)
        print("Points (Birth, Death) for H0:\n", point_diagram_list[0])
    
        # save diagrams dict as pickle
        diag_filepath = open_write_file(DIAGRAM_POINT_DIRECTORY, f'{system}.pkl')
        with open(diag_filepath, 'wb') as f:
            pickle.dump(diagrams_with_id, f)

    print(f"------ END PERSISTENCE ------")


############### PERSISTENCE END ###############


if __name__ == "__main__":
    args = _parse_args()

    assert args.source in ['graph', 'point']
    assert sum([args.abs, args.plqy, args.plqy_full]) <= 1, "Can only choose one of abs/plqy/plqy-full"

    DATA_DIRECTORY = "data/pretrain"
    if args.abs:
        DATA_DIRECTORY = "data/abs"
    if args.plqy:
        DATA_DIRECTORY = "data/plqy"
    if args.plqy_full:
        DATA_DIRECTORY = "data/plqy-full"
    
    CIF_DIRECTORY = f"{DATA_DIRECTORY}/cif"
    GRAPH_DIRECTORY = f"{DATA_DIRECTORY}/graphs"
    STRUCTURE_DIRECTORY = f"{DATA_DIRECTORY}/structs"
    DIAGRAM_GRAPH_DIRECTORY = f"{DATA_DIRECTORY}/diagrams_g"
    DIAGRAM_POINT_DIRECTORY = f"{DATA_DIRECTORY}/diagrams_p"
    MAX_DIST = get_max_dist(DATA_DIRECTORY)

    if args.source == 'graph':
        compute_persistence_diagrams_graph()
    else:
        compute_persistence_diagrams_point(
            plqy=args.plqy, 
            plqy_full=args.plqy_full,
        )