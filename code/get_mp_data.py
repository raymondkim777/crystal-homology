import os
import json
from mp_api.client import MPRester
from monty.json import MontyEncoder
# from monty.serialization import loadfn, dumpfn
from dotenv import load_dotenv
load_dotenv()


CRYSTAL_SYSTEMS = [
    # 'triclinic', 
    # 'monoclinic', 
    # 'trigonal', 
    # 'hexagonal', 
    'orthorhombic', 
    'tetragonal', 
    'cubic'
]
FIELDS = [
    "material_id", 
    "symmetry", 
    "structure"
]


def open_write_file(dir_path, file_name):
    """Opens a file for writing, or creates new file if file doesn't exist."""
    file_path = os.path.join(dir_path, file_name)
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    return file_path


def query_all_crystals_from_mp():
    for system in CRYSTAL_SYSTEMS:
        print(f"Querying {system} system from Materials Project...")

        with MPRester() as mpr:
            docs = mpr.materials.summary.search(
                crystal_system=system,
                fields=FIELDS
            )

        # serialize SummaryDoc into JSON
        # ! Note: directly using Monty serialization messes the material_id
        mp_json_list = []
        for doc in docs:
            json_object = dict()
            for field in FIELDS:
                json_object[field] = getattr(doc, field)
            mp_json_list.append(json_object)
        
        # save JSON files
        data_raw_dir = f'data/mp-raw'
        file_raw_name = f'{system}.json'
        data_raw_path = open_write_file(data_raw_dir, file_raw_name)

        with open(data_raw_path, 'w') as f:
            json.dump(mp_json_list, f, cls=MontyEncoder, indent=4)

        
# with MPRester(monty_decode=False, use_document_model=False) as mpr:
#     pass


if __name__ == "__main__":
    query_all_crystals_from_mp()