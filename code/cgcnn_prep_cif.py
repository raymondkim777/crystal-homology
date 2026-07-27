import os
import argparse
import random
import pickle
import csv
import shutil
import numpy as np
import networkx as nx
import warnings
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed
from create_bonds import get_structures_from_cif
from vectorizers import IM_BANDWIDTH, IM_RESOLUTION, fit_image_transformers
from utils import CRYSTAL_SYSTEMS, DIMENSION_CNT, get_num_cpus, open_write_file
from utils import PREDICT, ABS_PREDICT, PLQY_PREDICT, TASK_SPECS, ABS_TASK_SPECS, PLQY_TASK_SPECS


DATA_PRE_DIRECTORY = "data/pretrain"
DATA_ABS_DIRECTORY = "data/abs"
DATA_PLQY_DIRECTORY = "data/plqy"

CGCNN_DATAPATH = 'cgcnn/data'
CGCNN_PRE_DATAPATH = f'{CGCNN_DATAPATH}/pretrain_c'
CGCNN_ABS_DATAPATH = f'{CGCNN_DATAPATH}/abs_c'
CGCNN_PLQY_DATAPATH = f'{CGCNN_DATAPATH}/plqy_c'


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--abs', action='store_true', help='Saves absorption data to CGCNN data path')
    parser.add_argument('--merge', action='store_true', help='Merges absorption data and subset data')
    parser.add_argument('--plqy', action='store_true', help='Saves PLQY data to CGCNN data path')
    parser.add_argument('--vector', action='store_true', help='Saves vectorizations to CGCNN data path')
    parser.add_argument('--bound', action='store_true', help='Saves persistence bounds to CGCNN data path')
    return parser.parse_args()


def fetch_cif_filenames(data_dir, system: str) -> list:
    print(f"Fetching CIF files of {system} system...")
    cif_files = []

    # os.scandir() returns an iterator of DirEntry objects
    with os.scandir(f"{data_dir}/cif/{system}") as entries:
        for entry in entries:
            if not entry.is_file():
                continue
            cif_files.append(entry.name)
    return cif_files


def unpack_system_cifs(data_dir, dest_dir, system):
    cif_files = fetch_cif_filenames(data_dir, system)
    for file in cif_files:
        shutil.copy(f"{data_dir}/cif/{system}/{file}", dest_dir)

    # # data_dir, system = args
    # with open(f'{data_dir}/structs/{system}.pkl', 'rb') as file:
    #     struct_system_dict = pickle.load(file)

    # for cif_id, struct in struct_system_dict.items():
    #     struct_system_dict[cif_id] = {
    #         'structure': struct,
    #         'system': system, 
    #         'lattice_matrix': struct.lattice.matrix,
    #     }
    
    # return system, struct_system_dict


def unpack_all_cifs(data_dir, dest_dir) -> dict:
    num_workers = get_num_cpus()
    open_write_file(dest_dir, '')
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(unpack_system_cifs, data_dir, dest_dir, system): system
            for system in CRYSTAL_SYSTEMS
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc='Unpacking systems'):
            future.result()


def write_pretrain_id_prop_mask(abs=False, merge=False):
    print(f"\nPreparing PRETRAIN prop/mask CSVs...")
    doc_dict = dict()
    for system in CRYSTAL_SYSTEMS:
        print(f"Unpacking pretrain {system} system docs...")
        with open(f'{DATA_PRE_DIRECTORY}/mp-subset/{system}.pkl', 'rb') as file:
            system_dict = pickle.load(file)
        doc_dict.update(system_dict)
    
    csv_prop_data = []
    csv_mask_data = []
    system_to_int = {CRYSTAL_SYSTEMS[idx]: idx for idx in range(len(CRYSTAL_SYSTEMS))}

    print(f"\nComputing PRETRAIN{'/ABS' if abs and merge else ''} id_prop.csv and id_mask.csv...")
    for mp_id, doc in tqdm(doc_dict.items()):
        prop_dict = {
            'system': system_to_int[str(doc['symmetry'].crystal_system).lower()],
            # 'bm_voigt': doc['bulk_modulus']['voigt'] if doc['bulk_modulus'] is not None else 0.0,
            # 'bm_reuss': doc['bulk_modulus']['reuss'] if doc['bulk_modulus'] is not None else 0.0,
            # 'bm_vrh': doc['bulk_modulus']['vrh'] if doc['bulk_modulus'] is not None else 0.0,
            'direct_gap': doc['bandstructure'].latimer_munro.direct_gap 
            if doc['bandstructure'] is not None and doc['bandstructure'].latimer_munro is not None 
            else 0.0,
            'band_gap': doc['band_gap'] if doc['band_gap'] is not None else 0.0,
            'efermi': doc['efermi'] if doc['efermi'] is not None else 0.0,
            'is_gap_direct': 1 if doc['is_gap_direct'] is not None and doc['is_gap_direct'] else 0,
        }
        mask_dict = {
            'system': 1,
            # 'bm_voigt': int(doc['bulk_modulus'] is not None),
            # 'bm_reuss': int(doc['bulk_modulus'] is not None),
            # 'bm_vrh': int(doc['bulk_modulus'] is not None),
            'direct_gap': int(doc['bandstructure'] is not None and doc['bandstructure'].latimer_munro is not None),
            'band_gap': int(doc['band_gap'] is not None),
            'efermi': int(doc['efermi'] is not None),
            'is_gap_direct': int(doc['is_gap_direct'] is not None),
        }
        if abs and merge:
            if 'absorption' in doc.keys():
                doc_abs = doc['absorption']
                prop_dict.update({
                    'max_absorption': doc_abs['max_absorption'] if doc_abs['max_absorption'] is not None else 0.0,
                    'max_absorption_energy': doc_abs['max_absorption_energy'] if doc_abs['max_absorption_energy'] is not None else 0.0,
                    'integrated_absorption': doc_abs['integrated_absorption'] if doc_abs['integrated_absorption'] is not None else 0.0,
                    'integrated_absorption_visible': doc_abs['integrated_absorption_visible'] if doc_abs['integrated_absorption_visible'] is not None else 0.0,
                    'average_absorption_visible': doc_abs['average_absorption_visible'] if doc_abs['average_absorption_visible'] is not None else 0.0,
                    'absorption_onset_energy': doc_abs['absorption_onset_energy'] if doc_abs['absorption_onset_energy'] is not None else 0.0,
                })
            else:
                prop_dict.update({
                    'max_absorption': 0.0,
                    'max_absorption_energy': 0.0,
                    'integrated_absorption': 0.0,
                    'integrated_absorption_visible': 0.0,
                    'average_absorption_visible': 0.0,
                    'absorption_onset_energy': 0.0,
                })
            if 'absorption' in doc.keys():
                doc_abs = doc['absorption']
                mask_dict.update({
                    'max_absorption': int(doc_abs['max_absorption'] is not None),
                    'max_absorption_energy': int(doc_abs['max_absorption_energy'] is not None),
                    'integrated_absorption': int(doc_abs['integrated_absorption'] is not None),
                    'integrated_absorption_visible': int(doc_abs['integrated_absorption_visible'] is not None),
                    'average_absorption_visible': int(doc_abs['average_absorption_visible'] is not None),
                    'absorption_onset_energy': int(doc_abs['absorption_onset_energy'] is not None),
                })
            else:
                mask_dict.update({
                    'max_absorption': 0,
                    'max_absorption_energy': 0,
                    'integrated_absorption': 0,
                    'integrated_absorption_visible': 0,
                    'average_absorption_visible': 0,
                    'absorption_onset_energy': 0,
                })
        csv_prop_row = [mp_id[3:]]
        csv_mask_row = [mp_id[3:]]

        # ensure order is same as PREDICT in utils.py
        for prop in PREDICT:
            csv_prop_row.append(prop_dict[prop])
            csv_mask_row.append(mask_dict[prop])
        csv_prop_data.append(csv_prop_row)
        csv_mask_data.append(csv_mask_row)
    
    print(f"\nWriting PRETRAIN id_prop.csv and id_mask.csv...")
    csv_prop_filepath = open_write_file(CGCNN_PRE_DATAPATH, 'id_prop.csv')
    with open(csv_prop_filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(csv_prop_data)

    csv_mask_filepath = open_write_file(CGCNN_PRE_DATAPATH, 'id_mask.csv')
    with open(csv_mask_filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(csv_mask_data)


def write_abs_id_prop_mask():
    print(f"\nPreparing ABS prop/mask CSVs...")
    doc_dict = dict()
    for system in CRYSTAL_SYSTEMS:
        print(f"Unpacking ABS {system} system docs...")
        with open(f'{DATA_ABS_DIRECTORY}/mp-abs/{system}.pkl', 'rb') as file:
            system_dict = pickle.load(file)
        doc_dict.update(system_dict)
    
    csv_prop_data = []
    csv_mask_data = []

    print(f"\nComputing ABS id_prop.csv and id_mask.csv...")
    for mp_id, doc_abs in tqdm(doc_dict.items()):
        prop_dict = {
            'max_absorption': doc_abs['max_absorption'] if doc_abs['max_absorption'] is not None else 0.0,
            'max_absorption_energy': doc_abs['max_absorption_energy'] if doc_abs['max_absorption_energy'] is not None else 0.0,
            'integrated_absorption': doc_abs['integrated_absorption'] if doc_abs['integrated_absorption'] is not None else 0.0,
            'integrated_absorption_visible': doc_abs['integrated_absorption_visible'] if doc_abs['integrated_absorption_visible'] is not None else 0.0,
            'average_absorption_visible': doc_abs['average_absorption_visible'] if doc_abs['average_absorption_visible'] is not None else 0.0,
            'absorption_onset_energy': doc_abs['absorption_onset_energy'] if doc_abs['absorption_onset_energy'] is not None else 0.0,
        }
        mask_dict = {
            'max_absorption': int(doc_abs['max_absorption'] is not None),
            'max_absorption_energy': int(doc_abs['max_absorption_energy'] is not None),
            'integrated_absorption': int(doc_abs['integrated_absorption'] is not None),
            'integrated_absorption_visible': int(doc_abs['integrated_absorption_visible'] is not None),
            'average_absorption_visible': int(doc_abs['average_absorption_visible'] is not None),
            'absorption_onset_energy': int(doc_abs['absorption_onset_energy'] is not None),
        }
        csv_prop_row = [mp_id[3:]]
        csv_mask_row = [mp_id[3:]]

        # ensure order is same as ABS_PREDICT in utils.py
        for prop in ABS_PREDICT:
            csv_prop_row.append(prop_dict[prop])
            csv_mask_row.append(mask_dict[prop])
        csv_prop_data.append(csv_prop_row)
        csv_mask_data.append(csv_mask_row)
    
    print(f"\nWriting ABS id_prop.csv and id_mask.csv...")
    csv_prop_filepath = open_write_file(CGCNN_ABS_DATAPATH, 'id_prop.csv')
    with open(csv_prop_filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(csv_prop_data)

    csv_mask_filepath = open_write_file(CGCNN_ABS_DATAPATH, 'id_mask.csv')
    with open(csv_mask_filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(csv_mask_data)


def write_plqy_id_prop_mask():
    print(f"\nPreparing PLQY prop/mask CSVs...")
    doc_dict = dict()
    for system in CRYSTAL_SYSTEMS:
        print(f"Unpacking PLQY {system} system docs...")
        with open(f'{DATA_PLQY_DIRECTORY}/mp-plqy/{system}.pkl', 'rb') as file:
            system_dict = pickle.load(file)
        doc_dict.update(system_dict)
    
    csv_prop_data = []
    csv_mask_data = []

    print(f"\nComputing PLQY id_prop.csv and id_mask.csv...")
    for mp_id, doc in tqdm(doc_dict.items()):
        prop_dict = {
            'plqy': doc['plqy'],
        }
        mask_dict = {
            'plqy': 1,
        }
        csv_prop_row = [mp_id]  # id names don't have 'mp-'
        csv_mask_row = [mp_id]

        # ensure order is same as ABS_PREDICT in utils.py
        for prop in PLQY_PREDICT:
            csv_prop_row.append(prop_dict[prop])
            csv_mask_row.append(mask_dict[prop])
        csv_prop_data.append(csv_prop_row)
        csv_mask_data.append(csv_mask_row)
    
    print(f"\nWriting PLQY id_prop.csv and id_mask.csv...")
    csv_prop_filepath = open_write_file(CGCNN_PLQY_DATAPATH, 'id_prop.csv')
    with open(csv_prop_filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(csv_prop_data)

    csv_mask_filepath = open_write_file(CGCNN_PLQY_DATAPATH, 'id_mask.csv')
    with open(csv_mask_filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(csv_mask_data)


def move_process(abs=False, merge=False, plqy=False, vector=False):
    """
    Saves all structures with labeled crystal systems in CGCNN data folder. 
    Computes and prints required bounds for GraphData.
    Structure structure:
    mp_id: {
        structure: <structure>, 
        system: <system>,
        lattice_matrix: <structure.lattice.matrix>,
    }
    Optionally saves diagrams and vectorizations as pickle files in CGCNN data folder. 
    """
    # save pretrain data
    print(f"Saving PRETRAIN CIF files to {CGCNN_PRE_DATAPATH}")
    unpack_all_cifs(DATA_PRE_DIRECTORY, CGCNN_PRE_DATAPATH)

    # save abs data if needed
    if abs:
        print(f"Saving ABS structure files to {CGCNN_ABS_DATAPATH}")
        unpack_all_cifs(DATA_ABS_DIRECTORY, CGCNN_ABS_DATAPATH)

    # save PLQY data if needed
    if plqy:
        print(f"Saving PLQY structure files to {CGCNN_PLQY_DATAPATH}")
        print(f"\nUnpacking PLQY structures...")
        unpack_all_cifs(DATA_PLQY_DIRECTORY, CGCNN_PLQY_DATAPATH)
    
    # multitask regression/classification id_prop and id_mask
    write_pretrain_id_prop_mask(abs=abs, merge=merge)

    print(f"\nSaving PRETRAIN task specs...")
    # save PREDICT list to CGCNN data path
    predict_filepath = open_write_file(f"{CGCNN_PRE_DATAPATH}/tasks", f'predict.pkl')
    with open(predict_filepath, 'wb') as f:
        pickle.dump(PREDICT, f)

    # save TASK_SPEC dict to CGCNN data path
    task_filepath = open_write_file(f"{CGCNN_PRE_DATAPATH}/tasks", f'tasks.pkl')
    with open(task_filepath, 'wb') as f:
        pickle.dump(TASK_SPECS, f)

    # copy atom_init.json to CGCNN data path
    source_file = f'{CGCNN_DATAPATH}/atom_init.json'
    destination = open_write_file(f'{CGCNN_PRE_DATAPATH}', '')
    shutil.copy(source_file, destination)

    if abs and not merge:
        # multitask regression/classification id_prop and id_mask
        write_abs_id_prop_mask()

        print(f"\nSaving ABS task specs...")
        # save ABS_PREDICT list to CGCNN data path
        predict_filepath = open_write_file(f"{CGCNN_ABS_DATAPATH}/tasks", f'predict.pkl')
        with open(predict_filepath, 'wb') as f:
            pickle.dump(ABS_PREDICT, f)

        # save ABS_TASK_SPEC dict to CGCNN data path
        task_filepath = open_write_file(f"{CGCNN_ABS_DATAPATH}/tasks", f'tasks.pkl')
        with open(task_filepath, 'wb') as f:
            pickle.dump(ABS_TASK_SPECS, f)

        # copy atom_init.json to CGCNN data path
        source_file = f'{CGCNN_DATAPATH}/atom_init.json'
        destination = open_write_file(f'{CGCNN_ABS_DATAPATH}', '')
        shutil.copy(source_file, destination)
    
    if plqy:
        # plqy id_prop and id_mask
        write_plqy_id_prop_mask()

        print(f"\nSaving PLQY task specs...")
        # save PLQY_PREDICT list to CGCNN data path
        predict_filepath = open_write_file(f"{CGCNN_PLQY_DATAPATH}/tasks", f'predict.pkl')
        with open(predict_filepath, 'wb') as f:
            pickle.dump(PLQY_PREDICT, f)

        # save ABS_TASK_SPEC dict to CGCNN data path
        task_filepath = open_write_file(f"{CGCNN_PLQY_DATAPATH}/tasks", f'tasks.pkl')
        with open(task_filepath, 'wb') as f:
            pickle.dump(PLQY_TASK_SPECS, f)

        # copy atom_init.json to CGCNN data path
        source_file = f'{CGCNN_DATAPATH}/atom_init.json'
        destination = open_write_file(f'{CGCNN_PLQY_DATAPATH}', '')
        shutil.copy(source_file, destination)


def save_bounds(abs=False, merge=False, plqy=False):
    print(f"\n Saving PRETRAIN image transformer bounds...")

    for ch in ['g', 'p']:
        source_file = f'{DATA_PRE_DIRECTORY}/image_bounds_{ch}.pkl'
        destination = open_write_file(f'{CGCNN_PRE_DATAPATH}/tasks', '')
        shutil.copy(source_file, destination)

        if abs and not merge:
            print(f"\n Saving ABS image transformer bounds...")
            source_file = f'{DATA_ABS_DIRECTORY}/image_bounds_{ch}.pkl'
            destination = open_write_file(f'{CGCNN_ABS_DATAPATH}/tasks', '')
            shutil.copy(source_file, destination)
        
        if plqy:
            print(f"\n Saving PLQY image transformer bounds...")
            source_file = f'{DATA_PLQY_DIRECTORY}/image_bounds_{ch}.pkl'
            destination = open_write_file(f'{CGCNN_PLQY_DATAPATH}/tasks', '')
            shutil.copy(source_file, destination)
            

if __name__ == "__main__":
    args = _parse_args()

    move_process(
        abs=args.abs, 
        merge=args.merge,
        plqy=args.plqy,
        vector=args.vector
    )
    if args.bound:
        save_bounds(abs=args.abs, merge=args.merge, plqy=args.plqy)