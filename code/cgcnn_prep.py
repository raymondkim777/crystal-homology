import os
import shutil
from tqdm import tqdm
import random
import csv
import numpy as np
import networkx as nx
from tqdm import tqdm
import pickle
from pymatgen.io.cif import CifParser
from utils import CRYSTAL_SYSTEMS, open_write_file
from utils import PREDICT, ABS_PREDICT, PLQY_PREDICT, TASK_SPECS, ABS_TASK_SPECS, PLQY_TASK_SPECS


DATA_PRE_DIRECTORY = "data/pretrain"
DATA_ABS_DIRECTORY = "data/abs"
DATA_PLQY_DIRECTORY = "data/plqy"

CGCNN_DATAPATH = 'cgcnn/data'
CGCNN_PRE_DATAPATH = f'{CGCNN_DATAPATH}/pretrain_base'
CGCNN_ABS_DATAPATH = f'{CGCNN_DATAPATH}/abs_base'
CGCNN_PLQY_DATAPATH = f'{CGCNN_DATAPATH}/plqy_base'


def fetch_cif_filenames(cif_dir, system: str) -> list:
    print(f"Fetching CIF files of {system} system in {cif_dir}...")
    cif_files = []

    # os.scandir() returns an iterator of DirEntry objects
    with os.scandir(f"{cif_dir}/{system}") as entries:
        for entry in entries:
            if not entry.is_file():
                continue
            cif_files.append(entry.name)
    return cif_files


def write_pretrain_id_prop_mask():
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

    print(f"\nComputing PRETRAIN id_prop.csv and id_mask.csv...")
    # counter = 0
    # doc_dict_list = list(doc_dict.items())
    # random.shuffle(doc_dict_list)
    # for mp_id, doc in tqdm(doc_dict_list):
    for mp_id, doc in tqdm(doc_dict.items()):
        # if counter > 1000:
        #     break
        # counter += 1
        prop_dict = {
            'system': system_to_int[str(doc['symmetry'].crystal_system).lower()],
            'direct_gap': doc['bandstructure'].latimer_munro.direct_gap 
            if doc['bandstructure'] is not None and doc['bandstructure'].latimer_munro is not None 
            else 0.0,
            'band_gap': doc['band_gap'] if doc['band_gap'] is not None else 0.0,
            'efermi': doc['efermi'] if doc['efermi'] is not None else 0.0,
            'is_gap_direct': 1 if doc['is_gap_direct'] is not None and doc['is_gap_direct'] else 0,
        }
        mask_dict = {
            'system': 1,
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


def prep_data() -> dict:
    open_write_file(CGCNN_PRE_DATAPATH, '')
    for system in CRYSTAL_SYSTEMS:
        cif_files = fetch_cif_filenames(f"{DATA_PRE_DIRECTORY}/cif", system)
                    
        print(f"Moving CIF files of {system} system...")
        for filename in tqdm(cif_files):
            cif_filename = f"{DATA_PRE_DIRECTORY}/cif/{system}/{filename}"
            
            # copy file over to cgcnn data folder
            shutil.copy(cif_filename, f"{CGCNN_PRE_DATAPATH}/{filename[3:]}")
    
    # create id_prop.csv
    write_pretrain_id_prop_mask()

    print(f"\nSaving PRETRAIN task specs...")
    # save PREDICT list to CGCNN data path
    predict_filepath = open_write_file(f"{CGCNN_PRE_DATAPATH}/tasks", f'predict.pkl')
    with open(predict_filepath, 'wb') as f:
        pickle.dump(PREDICT, f)

    # save TASK_SPEC dict to CGCNN data path
    task_filepath = open_write_file(f"{CGCNN_PRE_DATAPATH}/tasks", f'tasks.pkl')
    with open(task_filepath, 'wb') as f:
        pickle.dump(TASK_SPECS, f)

    # save max nbrs and bond dist info to CGCNN data path
    source_file = f'{DATA_PRE_DIRECTORY}/bounds.json'
    destination = open_write_file(f'{CGCNN_PRE_DATAPATH}/tasks', '')
    shutil.copy(source_file, destination)

    # copy atom_init.json to CGCNN data path
    source_file = f'{CGCNN_DATAPATH}/atom_init.json'
    destination = open_write_file(f'{CGCNN_PRE_DATAPATH}', '')
    shutil.copy(source_file, destination)




    open_write_file(CGCNN_ABS_DATAPATH, '')
    for system in CRYSTAL_SYSTEMS:
        cif_files = fetch_cif_filenames(f"{DATA_ABS_DIRECTORY}/cif", system)
                    
        print(f"Moving CIF files of {system} system...")
        for filename in tqdm(cif_files):
            cif_filename = f"{DATA_ABS_DIRECTORY}/cif/{system}/{filename}"
            
            # copy file over to cgcnn data folder
            shutil.copy(cif_filename, f"{CGCNN_ABS_DATAPATH}/{filename[3:]}")

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

    # save max nbrs and bond dist info to CGCNN data path
    source_file = f'{DATA_ABS_DIRECTORY}/bounds.json'
    destination = open_write_file(f'{CGCNN_ABS_DATAPATH}/tasks', '')
    shutil.copy(source_file, destination)

    # copy atom_init.json to CGCNN data path
    source_file = f'{CGCNN_DATAPATH}/atom_init.json'
    destination = open_write_file(f'{CGCNN_ABS_DATAPATH}', '')
    shutil.copy(source_file, destination)
    


    open_write_file(CGCNN_PLQY_DATAPATH, '')
    for system in CRYSTAL_SYSTEMS:
        cif_files = fetch_cif_filenames(f"{DATA_PLQY_DIRECTORY}/cif", system)
                    
        print(f"Moving CIF files of {system} system...")
        for filename in tqdm(cif_files):
            cif_filename = f"{DATA_PLQY_DIRECTORY}/cif/{system}/{filename}"
            
            # copy file over to cgcnn data folder
            shutil.copy(cif_filename, f"{CGCNN_PLQY_DATAPATH}/{filename}")
    
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

        # save max nbrs and bond dist info to CGCNN data path
    source_file = f'{DATA_PLQY_DIRECTORY}/bounds.json'
    destination = open_write_file(f'{CGCNN_PLQY_DATAPATH}/tasks', '')
    shutil.copy(source_file, destination)

    # copy atom_init.json to CGCNN data path
    source_file = f'{CGCNN_DATAPATH}/atom_init.json'
    destination = open_write_file(f'{CGCNN_PLQY_DATAPATH}', '')
    shutil.copy(source_file, destination)


if __name__ == "__main__":
    prep_data()
    # bid_test()
    # test()