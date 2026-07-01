import argparse
import os
import csv
import pickle
import shutil
import requests
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
import warnings

from pymatgen.io.cif import CifParser
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core.periodic_table import Element
from pymatgen.core.composition import Composition
from utils import CRYSTAL_SYSTEMS, open_write_file


COD_CSV_PATH = 'data/COD-selection.csv'
PLQY_FULL_PATH = 'data/plqy-full'
PLQY_FULL_CIF_PATH = f'{PLQY_FULL_PATH}/cif'
PLQY_FULL_CIF_RAW_PATH = f'{PLQY_FULL_PATH}/cif-raw'
PLQY_FULL_DOC_PATH = f'{PLQY_FULL_PATH}/cod-plqy'


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='prints debug statements to terminal')
    return parser.parse_args()


def is_cu_halide_in_str(string):
    string = string.lower()
    if 'cu' not in string:
        return False
    halides = ['f', 'cl', 'br', 'i']
    for hal in halides:
        if hal in string:
            return True
    return False


def filter_copper_and_halide():
    cif_ids = []
    with open(COD_CSV_PATH, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for _ in range(11):
            next(reader)
        for row in reader:
            if is_cu_halide_in_str(row[33]):
                cif_ids.append(row[0])
    return cif_ids


def collect_cifs(id_list):
    for mp_id in id_list:
        link = f'http://crystallography.net/cod/{mp_id}.cif'
        file_name = f'cod-{mp_id}.cif'
        response = requests.get(link, stream=True)

        if response.status_code == 200:
            file_path = open_write_file(PLQY_FULL_CIF_RAW_PATH, file_name)
            with open(file_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print(f"Downloaded and saved {file_name} from COD")
        else:
            print(f"Failed to download {file_name} from COD")


def fetch_cif_filenames() -> list:
    print(f"Fetching CIF files for PLQY full...")
    cif_files = []

    # os.scandir() returns an iterator of DirEntry objects
    with os.scandir(f"{PLQY_FULL_CIF_RAW_PATH}") as entries:
        for entry in entries:
            if not entry.is_file():
                continue
            cif_files.append(entry.name)
    return cif_files


def get_structures_from_cif(filepath: str) -> list:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cif_parser = CifParser(filepath, occupancy_tolerance=1.1)
        structures = cif_parser.parse_structures()
    
    if len(structures) > 1:
        print(f"[PLQY Structures] File {filepath} generates multiple structures")
    
    # for struct in structures:
    #     check_result = cif_parser.check(struct)
    #     if check_result is not None:
    #         print(f"CIF Error: {filepath}")
    #         print(f"Error Message: {check_result}")
    #         raise ValueError(f"Struct contained in {filepath} is invalid")
    return structures


def define_node_species(structure, nx_multigraph):
    # nx_multigraph is mutable
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


def extract_inorganic_subgraph(nx_multigraph):
    # filter nodes with cu or halide
    cu_hal_nums = [29, 9, 17, 35, 53]
    induce_nodes = []
    for node in nx_multigraph.nodes:
        node_species = nx_multigraph.nodes[node]['species']
        cu_hal_exist = False
        for at_num in cu_hal_nums:
            if at_num in node_species.keys():
                cu_hal_exist = True
        if cu_hal_exist:
            induce_nodes.append(node)
    
    # extract induced subgraph of copper halides
    cu_hal_subgraph = nx_multigraph.subgraph(induce_nodes)  # .copy() ??
    return cu_hal_subgraph, induce_nodes


def is_cu_halide_edge(attr1, attr2):
    cu_num = 29
    halide_nums = [9, 17, 35, 53]
    if cu_num in attr1['species'].keys():
        for hal_num in halide_nums:
            if hal_num in attr2['species'].keys():
                return True
    if cu_num in attr2['species'].keys():
        for hal_num in halide_nums:
            if hal_num in attr1['species'].keys():
                return True
    return False


def is_organic_edge(attr1, attr2):
    c_num, h_num = 6, 1
    if c_num in attr1['species'].keys():
        if c_num in attr2['species'].keys():
            return True
        if h_num in attr2['species'].keys():
            return True
    if c_num in attr2['species'].keys():
        if h_num in attr1['species'].keys():
            return True
    return False


def collect_hybrid_cu_halides(debug=False):
    if not os.path.isdir(PLQY_FULL_CIF_RAW_PATH):
        id_list = filter_copper_and_halide()
        print(f'Crystal # with Cu and Halides: {len(id_list)}')
        collect_cifs(id_list)
    else:
        print(f"PLQY full CIF directory exists, skipping CIF downloads")
    
    cif_filenames = fetch_cif_filenames()

    crystalnn = CrystalNN(
        distance_cutoffs=None, 
        x_diff_weight=0, 
        porous_adjustment=False,
        search_cutoff=11,  # default 7, but ERROR: No Voronoi neighbors found for site
    )
    
    open_write_file(PLQY_FULL_DOC_PATH, '')
    plqy_docs_dict = dict()
    for system in CRYSTAL_SYSTEMS:
        open_write_file(f"{PLQY_FULL_CIF_PATH}/{system}", '')

    for filename in tqdm(cif_filenames, desc='Parsing CIFs; '):
        if debug:
            print(f"CIF: {filename}")
            print("Retrieving Pymatgen Structure from CIF")
        
        # retrieve pymatgen structure
        structure_filename = f"{PLQY_FULL_CIF_RAW_PATH}/{filename}"
        structures = get_structures_from_cif(structure_filename)
        
        if debug:
            print("Getting graph")
        
        # get graph from structure
        structure = structures[0].get_reduced_structure()
        bonded_structure = crystalnn.get_bonded_structure(
            structure,
            # on_disorder='take_majority_strict', 
            on_disorder='take_max_species', 
        )
        nx_multigraph = bonded_structure.graph

        if debug:
            print("Getting node species")
        
        # fix node species information
        define_node_species(structure, nx_multigraph)

        if debug:
            print("Getting inorganic subgraph")
        
        # get inorganic subgraph
        cu_hal_subgraph, subgraph_nodes = extract_inorganic_subgraph(nx_multigraph)
        
        if debug:
            print("Test node attr from inorganic subgraph:")
            print(cu_hal_subgraph.nodes[subgraph_nodes[0]]['species'])
        
        # check if edge exists b/w cu & halide
        inorganic_exist = False
        for u, v in cu_hal_subgraph.edges():
            attr1 = cu_hal_subgraph.nodes[u]
            attr2 = cu_hal_subgraph.nodes[v]
            if is_cu_halide_edge(attr1, attr2):
                inorganic_exist = True
                break
        
        if not inorganic_exist:
            if debug:
                print("No Cu-halide edge in inorganic")
            continue
            
        if debug:
            print("YES Cu-halide edge in inorganic")
        
        if debug:
            print("Geting remaining organic? subgraph")

        # get remaining graph (potential organic portion)
        remaining_nodes = [n for n in nx_multigraph.nodes() if n not in subgraph_nodes]
        remaining_subgraph = nx_multigraph.subgraph(remaining_nodes)

        if debug:
            print("Test node attr from organic? subgraph:")
            if len(remaining_nodes) > 0:
                print(remaining_subgraph.nodes[remaining_nodes[0]]['species'])
            else:
                print("No organic portion")

        # check if C-C or C-H edge exists
        organic_exist = False
        for u, v in remaining_subgraph.edges():
            attr1 = remaining_subgraph.nodes[u]
            attr2 = remaining_subgraph.nodes[v]
            if is_organic_edge(attr1, attr2):
                organic_exist = True
                break

        if organic_exist:
            if debug:
                print("YES organic edge found")
            # save docs and CIF
            crystal_id = filename[4:-4]
            plqy_docs_dict[crystal_id] = {
                'structure': structure,
            }
            shutil.copy(structure_filename, f"{PLQY_FULL_CIF_PATH}/{CRYSTAL_SYSTEMS[0]}")
    
    print(f"Found {len(plqy_docs_dict.keys())} hybrid Cu halides")

    # save all under cubic system; meaningless, only done for compatibility
    print(f"Saving PLQY full docs under {CRYSTAL_SYSTEMS[0]} system...")
    subset_plqy_path = open_write_file(PLQY_FULL_DOC_PATH, f'{CRYSTAL_SYSTEMS[0]}.pkl')
    with open(subset_plqy_path, 'wb') as f:
        pickle.dump(plqy_docs_dict, f)
    
    # fill other systems with empty dict
    print(f"Filling all other systems with empty PLQY docs...")
    empty_dict = dict()
    for system in CRYSTAL_SYSTEMS[1:]:
        subset_plqy_path = open_write_file(PLQY_FULL_DOC_PATH, f'{system}.pkl')
        with open(subset_plqy_path, 'wb') as f:
            pickle.dump(empty_dict, f)


if __name__ == '__main__':
    args = _parse_args()
    collect_hybrid_cu_halides(debug=args.debug)