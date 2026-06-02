import json
import random
from monty.json import MontyEncoder
from monty.serialization import loadfn
from utils import open_write_file
from dotenv import load_dotenv
load_dotenv()


CRYSTAL_SYSTEMS = [
    'triclinic', 
    'monoclinic', 
    'trigonal', 
    'hexagonal', 
    'orthorhombic', 
    'tetragonal', 
    'cubic'
]
MP_DATA_PATH = 'data/mp-raw'
SUBSET_SIZE = 1000


def choose_subset_of_mp_data():
    for system in CRYSTAL_SYSTEMS:
        print(f"Choosing subset of {system} system...")
        mp_raw_json_list = loadfn(f"data/mp-raw/{system}.json")
        mp_subset = random.sample(mp_raw_json_list, k=SUBSET_SIZE)

        # save subset JSON files
        data_raw_dir = f'data/mp-subset'
        file_raw_name = f'{system}.json'
        data_raw_path = open_write_file(data_raw_dir, file_raw_name)

        with open(data_raw_path, 'w') as f:
            json.dump(mp_subset, f, cls=MontyEncoder, indent=4)


def convert_summarydoc_to_cif():
    for system in CRYSTAL_SYSTEMS:
        print(f"Converting {system} system files into CIF...")
        data_cif_dir = f'data/cif/{system}'
        mp_json_list = loadfn(f"data/mp-subset/{system}.json")

        for doc in mp_json_list:
            file_cif_name = f'{doc["material_id"]}.cif'
            data_cif_path = open_write_file(data_cif_dir, file_cif_name)

            cif_data = doc["structure"].to(fmt="cif")
            with open(data_cif_path, 'w') as f:
                f.write(cif_data)


if __name__ == "__main__":
    random.seed(42)
    choose_subset_of_mp_data()
    convert_summarydoc_to_cif()
