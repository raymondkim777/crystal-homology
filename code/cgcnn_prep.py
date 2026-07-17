import os
import shutil
from tqdm import tqdm
import random
import numpy as np
import networkx as nx
from tqdm import tqdm
import pickle
from pymatgen.io.cif import CifParser
from utils import CRYSTAL_SYSTEMS, open_write_file, plot_nxgraph


CIF_DIRECTORY = "data/cif"
CGCNN_DATAPATH = 'cgcnn/data/our-data'


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


def prep_data() -> dict:
    print(f"Saving files to {CGCNN_DATAPATH}")
    open_write_file(CGCNN_DATAPATH, '')  # creates cgcnn data directory

    id_to_system = dict()

    for system in CRYSTAL_SYSTEMS:
        cif_files = fetch_cif_filenames(system)
                    
        print(f"Moving CIF files of {system} system...")
        for filename in tqdm(cif_files):
            structure_filename = f"{CIF_DIRECTORY}/{system}/{filename}"
            
            # copy file over to cgcnn data folder
            shutil.copy(structure_filename, f"{CGCNN_DATAPATH}/{filename[3:]}")

            # store crystal system (for id_prop.csv)
            id_to_system[filename[:-4]] = system

    
    # create id_prop.csv
    system_to_int = {CRYSTAL_SYSTEMS[idx]: idx for idx in range(len(CRYSTAL_SYSTEMS))}
    
    csv_filepath = open_write_file(CGCNN_DATAPATH, 'id_prop.csv')
    with open(csv_filepath, 'w') as f:
        # CRYSTAL SYSTEM - classification
        for mp_id, system in id_to_system.items():
            f.write(f"{mp_id[3:]}, {system_to_int[system]}\n")



if __name__ == "__main__":
    prep_data()
    # bid_test()
    # test()