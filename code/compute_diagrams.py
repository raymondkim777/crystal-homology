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
from pymatgen.core.structure import PeriodicNeighbor

from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import CRYSTAL_SYSTEMS, DIMENSION_CNT, ATOM_DIMENSION_CNT
from utils import get_num_cpus, open_write_file, get_max_dist


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
ATOM_DIAGRAM_POINT_DIRECTORY = None
MAX_DIST = None
LABEL = None


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='print debug statements')
    parser.add_argument('--source', type=str, default='none', help="how to create complex [none, graph, point, custom]")
    parser.add_argument('--abs', action='store_true', help='constructs diagrams for absorption data')
    parser.add_argument('--plqy', action='store_true', help='constructs diagrams for plqy data')
    parser.add_argument('--plqy-full', action='store_true', help='constructs diagrams for plqy-full data')
    parser.add_argument('--atom', action='store_true', help='computes atom-specific persistent diagrams')
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


def plot_persistence_diagram_gtda(diagram, mp_id: str='') -> None:
    fig = plot_diagram(diagram)
    fig.show()
    # fig.save(f"{DIAGRAM_GRAPH_DIRECTORY}/diagram_{mp_id}.png")


def compute_persistence_diagrams_graph() -> None:
    print(f"------ {{{LABEL}}} GRAPH PERSISTENCE ------")
    n_jobs = get_num_cpus()
    print(f"Using {n_jobs} job processes")
    
    print(f"Unpacking all graphs...")
    graph_dict = unpack_all_graphs()

    flagser = FlagserPersistence(
        homology_dimensions=tuple(range(DIMENSION_CNT)),
        directed=True,
        # directed=False,
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
    '''
    distance matrix is normalized; max pairwise distance is 1
    '''
    # accept one tuple for multiprocessing
    system, filename, structure, plqy, plqy_full = args

    if structure is None:
        # print(f"extracting structure for {filename}")
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

    # print("Starting")
    # get distance matrix for each pair of frac. coordinates
    n = len(structure)
    dist_mat = np.zeros((n, n))
    if n < 2:
        return filename[:-4], dist_mat, 1

    max_dist = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist, jimage = structure.lattice.get_distance_and_image(
                structure.frac_coords[i], 
                structure.frac_coords[j],
            )
            dist_mat[i, j] = dist
            dist_mat[j, i] = dist
            max_dist = max(max_dist, dist)
    if max_dist == 0:
        max_dist = 1
    # print("Finished")

    return filename[:-4], dist_mat / max_dist, max_dist   # normalized to 1


def cif_to_dist_mat_norm(plqy=False, plqy_full=False):
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

        # uncompleted_files = set([filename[:-4] for filename in cif_files])

        with ProcessPoolExecutor(
            max_workers=n_workers,
        ) as executor:
            futures = [
                executor.submit(compute_dist_mat_for_cif, task)
                for task in tasks
            ]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"{system}: ",
            ):
                crystal_id, dist_mat, max_dist = future.result()
                system_dist_mats[crystal_id] = (dist_mat, max_dist)
                # uncompleted_files.remove(crystal_id)
                # print(uncompleted_files)

        # with ProcessPoolExecutor(
        #     max_workers=n_workers, 
        # ) as executor:
        #     results = executor.map(compute_dist_mat_for_cif, tasks)

        #     for crystal_id, dist_mat in tqdm(results, total=len(tasks), desc=f"PC for {system}: "):
        #         system_dist_mats[crystal_id] = dist_mat
        
        dist_dict[system] = system_dist_mats
    return dist_dict


def compute_persistence_diagrams_point(plqy=False, plqy_full=False):
    print(f"------ {{{LABEL}}} POINT CLOUD PERSISTENCE ------")
    n_workers = get_num_cpus()
    print(f"Using {n_workers} job processes")

    # compute normalized distance matrices for each crystal for each system
    dist_dict = cif_to_dist_mat_norm(plqy=plqy, plqy_full=plqy_full)

    rips = RipsPersistence(
        homology_dimensions=tuple(range(DIMENSION_CNT)),
        threshold=0.6, 
        input_type='full distance matrix', 
        n_jobs=n_workers,
    )

    # compute diagrams for each system
    for system in CRYSTAL_SYSTEMS:
        print(f"Computing PD for {system}")

        # dist_dict[system][mp_id] is (norm_dist_mat, max_dist)
        dist_mat_system = [item[0] for item in dist_dict[system].values()]
        max_dists_system = [item[1] for item in dist_dict[system].values()]
        # dist_mat_system = list(dist_dict[system].values())
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

        # multiply back into original distance scale, turn np.infs into MAX_DIST + 1
        assert len(diagrams) == len(max_dists_system)
        for crystal_diag, max_dist in zip(diagrams, max_dists_system):
            for dim in range(DIMENSION_CNT):
                # if crystal has 1 site, then PD will have NaN --> turn into inf, then MAX_DIST + 1
                # if max_dist == 0:
                #     print("MAX DISTANCE IS 0")
                crystal_diag[dim][np.isnan(crystal_diag[dim])] = np.inf
                crystal_diag[dim] *= max_dist
                crystal_diag[dim][np.isinf(crystal_diag[dim])] = MAX_DIST + 1

        # diagrams_filtered = []
        # diag_select = DiagramSelector(
        #     use=True, 
        #     point_type='finite', 
        #     limit=MAX_DIST,     # + 1??
        # )
        # for diag_crystal in diagrams:
        #     diagrams_filtered.append(diag_select.fit_transform(diag_crystal))

        # match diagrams to crystal id (format with multiple dimensions)
        keys = list(dist_dict[system].keys())
        diagrams_with_id = {
            keys[i]: diagrams[i]
            for i in range(len(diagrams))
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


############### CUSTOM POINT CLOUD PERSISTENCE ###############


def modify_dist_mat(dist_mat, graph):
    # directed graph (not multigraph)
    for i, j in graph.edges:
        dist_mat[i, j] = 0
        dist_mat[j, i] = 0
    return dist_mat


def compute_persistence_diagrams_custom(plqy=False, plqy_full=False):
    print(f"------ {{{LABEL}}} CUSTOM POINT CLOUD PERSISTENCE ------")
    n_workers = get_num_cpus()
    print(f"Using {n_workers} job processes")

    print(f"Unpacking all graphs...")
    graph_dict = unpack_all_graphs()

    # compute normalized distance matrices for each crystal for each system
    dist_dict = cif_to_dist_mat_norm(plqy=plqy, plqy_full=plqy_full)

    rips = RipsPersistence(
        homology_dimensions=tuple(range(DIMENSION_CNT)),
        threshold=0.6, 
        input_type='full distance matrix', 
        n_jobs=n_workers,
    )

    # compute diagrams for each system
    for system in CRYSTAL_SYSTEMS:
        print(f"Computing PD for {system}")

        # modify dist matrix s.t. graph edges have dist 0
        dist_mat_system = [
            modify_dist_mat(item[0], graph_dict[system][id])
            for id, item in dist_dict[system].items()
        ]
        max_dists_system = [item[1] for item in dist_dict[system].values()]
        # dist_mat_system = list(map(modify_dist_mat, dist_dict[system].values(), graph_dict[system].values()))

        if len(dist_mat_system) == 0:
            print(f"No crystals for {system} system!")
            diagrams_with_id = dict()
            diag_filepath = open_write_file(DIAGRAM_CUSTOM_DIRECTORY, f'{system}.pkl')
            with open(diag_filepath, 'wb') as f:
                pickle.dump(diagrams_with_id, f)
            continue
        
        # fit & transform rips to each crystal (point cloud)
        diagrams = rips.fit_transform(dist_mat_system)
        # [crystal1, crystal2, ...] where crystaln = [h0_diag, h1_diag, h2_diag]

        # multiply back into original distance scale, turn np.infs into MAX_DIST + 1
        assert len(diagrams) == len(max_dists_system)
        for crystal_diag, max_dist in zip(diagrams, max_dists_system):
            for dim in range(DIMENSION_CNT):
                crystal_diag[dim] *= max_dist
                crystal_diag[dim][np.isinf(crystal_diag[dim])] = MAX_DIST + 1

        # diagrams_filtered = []
        # diag_select = DiagramSelector(
        #     use=True, 
        #     point_type='finite', 
        #     limit=MAX_DIST,     # + 1??
        # )
        # for diag_crystal in diagrams:
        #     diagrams_filtered.append(diag_select.fit_transform(diag_crystal))

        # match diagrams to crystal id (format with multiple dimensions)
        keys = list(dist_dict[system].keys())
        diagrams_with_id = {
            keys[i]: diagrams[i]
            for i in range(len(diagrams))
        }

        # print diagram info
        print(f"diagram cnt: {len(diagrams)}")
        point_diagram_list = diagrams[0]
        print("Persistence Diagram Shape for H0:", point_diagram_list[0].shape)
        print("Points (Birth, Death) for H0:\n", point_diagram_list[0])
    
        # save diagrams dict as pickle
        diag_filepath = open_write_file(DIAGRAM_CUSTOM_DIRECTORY, f'{system}.pkl')
        with open(diag_filepath, 'wb') as f:
            pickle.dump(diagrams_with_id, f)

    print(f"------ END PERSISTENCE ------")


############### PERSISTENCE END ###############


################# ATOM START #################


def compute_atom_dist_mats_for_cif(args):
    '''
    distance matrix is normalized; max pairwise distance is 1
    '''
    # accept one tuple for multiprocessing
    system, filename, structure, plqy, plqy_full = args

    if structure is None:
        # print(f"extracting structure for {filename}")
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

    dist_mats_for_sites = [None for _ in structure]

    # iterate through each node
    for site_index, site in enumerate(structure):
        # site_index = structure.sites.index(site)
        radii = [3, 5, 7, 9]

        for radius in radii:
            # get neighborhood atoms within radius 
            neighborhood = structure.get_neighbors(site, radius)
            if len(neighborhood) <= int(len(structure) / 10):
                continue
            break

        # change neighborhood to indices list, add current site to neighborhood
        neighborhood = set([nbr.index for nbr in neighborhood])
        # print(f"num sites: {len(structure)}\nlen neighborhood: {len(neighborhood)}")
        # assert site_index not in neighborhood
        neighborhood.add(site_index)
        neighborhood = list(neighborhood)

        n = len(neighborhood)
        dist_mat = np.zeros((n, n))

        if n <= 1:
            dist_mats_for_sites[site_index] = (dist_mat, 1)
            continue

        max_dist = 0
        for i in range(n):
            for j in range(i + 1, n):
                idx_i = neighborhood[i]
                idx_j = neighborhood[j]

                dist, jimage = structure.lattice.get_distance_and_image(
                    structure.frac_coords[idx_i], 
                    structure.frac_coords[idx_j],
                )
                dist_mat[i, j] = dist
                dist_mat[j, i] = dist
                max_dist = max(max_dist, dist)
        if max_dist == 0:
            max_dist = 1
    
        dist_mats_for_sites[site_index] = (dist_mat / max_dist, max_dist)   # normalized to 1

    if DEBUG:
        print(f"END worker pid={os.getpid()} filename={filename}", flush=True)

    assert not None in dist_mats_for_sites, f"dist_mats_for_sites list has None item"
    return filename[:-4], dist_mats_for_sites


def cif_to_dist_mat_norm_atom(plqy=False, plqy_full=False):
    n_workers = get_num_cpus()
    dist_dict = dict()  # dist_dict[cif_id][node_idx] = distance matrix for local points

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

        system_dist_mats_sites = dict()     # dict[crystal_id][site_idx] = (normed_dist_mat, max_dist)

        cif_files = fetch_cif_filenames(system)
        if struct_exist:
            tasks = [(system, filename, system_structs[filename[:-4]], plqy, plqy_full) for filename in cif_files]
        else:
            tasks = [(system, filename, None, plqy, plqy_full) for filename in cif_files]

        # uncompleted_files = set([filename[:-4] for filename in cif_files])

        # with ProcessPoolExecutor(
        #     max_workers=n_workers,
        # ) as executor:
        #     futures = [
        #         executor.submit(compute_atom_dist_mats_for_cif, task)
        #         for task in tasks
        #     ]
        #     for future in tqdm(
        #         as_completed(futures),
        #         total=len(futures),
        #         desc=f"{system}: ",
        #     ):
        #         # crystal_id, dist_mats_for_sites = (normed_dist_mat, max_dist)
        #         print('start')
        #         crystal_id, dist_mats_for_sites = future.result()
        #         system_dist_mats_sites[crystal_id] = dist_mats_for_sites  # dict[crystal_id][site_idx] = (normed_dist_mat, max_dist)
        #         print("finish")

        #         # uncompleted_files.remove(crystal_id)
        #         # print(uncompleted_files)

        print("Computing atomic distance matrices for each crystal...")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(compute_atom_dist_mats_for_cif, task): task[1]
                for task in tasks
            }

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"{system}: ",
            ):
                filename = futures[future]

                if DEBUG:
                    print(f"START result: {filename}", flush=True)

                try:
                    crystal_id, dist_mats_for_sites = future.result()
                except Exception as e:
                    if DEBUG:
                        print(
                            f"FAILED: {filename}: {type(e).__name__}: {e}",
                            flush=True,
                        )
                        import traceback
                        traceback.print_exc()
                    continue

                system_dist_mats_sites[crystal_id] = dist_mats_for_sites

                if DEBUG:
                    print(f"FINISH result: {filename}", flush=True)

        dist_dict[system] = system_dist_mats_sites
    return dist_dict


def group2list(dist_groups):
    dist_list, max_dist_list = [], []
    for crystal_id, all_sites_dist in dist_groups.items():
        for site_dist, max_dist in all_sites_dist:
            dist_list.append(site_dist)
            max_dist_list.append(max_dist)
    return dist_list, max_dist_list


def list2group(dist_groups, pd_list):
    expected_cnt = sum(len(all_sites_dist) for all_sites_dist in dist_groups.values())
    if len(pd_list) != expected_cnt:
        raise ValueError(
            f"Expected {expected_cnt} persistence diagrams, "
            f"but received {len(pd_list)}."
        )
    
    pd_groups = dict()
    cur_idx = 0
    for crystal_id, all_sites_dist in dist_groups.items():
        site_cnt = len(all_sites_dist)
        crystal_pds = pd_list[cur_idx: cur_idx + site_cnt]
        pd_groups[crystal_id] = crystal_pds
        cur_idx += site_cnt
    return pd_groups


def compute_persistence_diagrams_point_atom():
    print(f"------ {{{LABEL}}} ATOM POINT CLOUD PERSISTENCE ------")
    n_workers = get_num_cpus()
    print(f"Using {n_workers} job processes")

    # compute normalized distance matrices for each crystal for each system
    dist_dict = cif_to_dist_mat_norm_atom(plqy=False, plqy_full=False)
    # ! dist_dict[cif_id][node_idx] = distance matrix for local points

    rips = RipsPersistence(
        homology_dimensions=tuple(range(ATOM_DIMENSION_CNT)),
        threshold=0.6, 
        input_type='full distance matrix', 
        n_jobs=n_workers,
    )

    # compute diagrams for each system
    for system in CRYSTAL_SYSTEMS:
        print(f"Computing PD for {system}")

        # dist_dict[system][crystal_id][site_idx] = (normed_dist_mat, max_dist)
        dist_mat_list, max_dist_list = group2list(dist_dict[system])

        if len(dist_mat_list) == 0:
            print(f"No crystals for {system} system!")
            diagrams_with_id = dict()
            diag_filepath = open_write_file(ATOM_DIAGRAM_POINT_DIRECTORY, f'{system}.pkl')
            with open(diag_filepath, 'wb') as f:
                pickle.dump(diagrams_with_id, f)
            continue

        # fit & transform rips to each atom point cloud
        diagrams = rips.fit_transform(dist_mat_list)

        # multiply back into original distance scale, turn np.infs into MAX_DIST + 1
        assert len(diagrams) == len(max_dist_list)
        for crystal_diag, max_dist in zip(diagrams, max_dist_list):
            for dim in range(DIMENSION_CNT):
                # if crystal has 1 site, then PD will have NaN --> turn into inf, then MAX_DIST + 1
                # if max_dist == 0:
                #     print("MAX DISTANCE IS 0")
                crystal_diag[dim][np.isnan(crystal_diag[dim])] = np.inf
                crystal_diag[dim] *= max_dist
                crystal_diag[dim][np.isinf(crystal_diag[dim])] = MAX_DIST + 1

        # match diagrams to crystal id (format with multiple dimensions)
        diagrams_with_id = list2group(dist_dict[system], diagrams)

        # print diagram info
        print(f"diagram cnt: {len(diagrams)}")
        point_diagram_list = diagrams[0]
        print("Persistence Diagram Shape for H0:", point_diagram_list[0].shape)
        print("Points (Birth, Death) for H0:\n", point_diagram_list[0])
    
        # save diagrams dict as pickle
        diag_filepath = open_write_file(ATOM_DIAGRAM_POINT_DIRECTORY, f'{system}.pkl')
        with open(diag_filepath, 'wb') as f:
            pickle.dump(diagrams_with_id, f)

    print(f"------ END ATOM PERSISTENCE ------")


################## ATOM END ##################


if __name__ == "__main__":
    args = _parse_args()
    DEBUG = args.debug

    assert args.source in ['none', 'graph', 'point', 'custom']
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
    DIAGRAM_CUSTOM_DIRECTORY = f"{DATA_DIRECTORY}/diagrams_c"
    ATOM_DIAGRAM_POINT_DIRECTORY = f"{DATA_DIRECTORY}/diagrams_atom_p"
    MAX_DIST = get_max_dist("data/pretrain")

    LABEL = 'ABS' if args.abs else 'PLQY' if args.plqy else 'PLQY-FULL' if args.plqy_full else 'PRETRAIN'

    if args.source != 'none':
        if args.source == 'graph':
            compute_persistence_diagrams_graph()
        elif args.source == 'point':
            compute_persistence_diagrams_point(
                plqy=args.plqy, 
                plqy_full=args.plqy_full,
            )
        else:
            compute_persistence_diagrams_custom(
                plqy=args.plqy, 
                plqy_full=args.plqy_full,
            )

    if args.atom:
        compute_persistence_diagrams_point_atom()