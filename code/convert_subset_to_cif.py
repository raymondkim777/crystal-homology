import argparse
import shutil
import json
import random
from tqdm import tqdm
from mp_api.client import MPRester
from monty.json import MontyEncoder
from monty.serialization import loadfn
from emmet.core.summary import HasProps
from utils import CRYSTAL_SYSTEMS, FIELDS, open_write_file

from dotenv import load_dotenv
load_dotenv()


MP_DATA_PATH = 'data/mp-raw'


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset-size', default=1000, type=int, help='size for each class')
    return parser.parse_args()


def clone_mp_raw():
    open_write_file('data/mp-base', '')
    for system in CRYSTAL_SYSTEMS:
        shutil.copyfile(f'data/mp-raw/{system}.json', f'data/mp-base/{system}.json')


def merge_abs_mp_data():
    with MPRester() as mpr:

        print(f"Querying summary data (absorption)...")
        # list_of_available_fields = mpr.materials.summary.available_fields
        # print(list_of_available_fields)
        docs = mpr.materials.summary.search(
            has_props=[HasProps.absorption],
            fields=FIELDS
        )

        docs_by_id = dict()
        mp_id_list_by_system = {system: [] for system in CRYSTAL_SYSTEMS}

        for doc in docs:
            system = str(doc.symmetry.crystal_system).lower()
            docs_by_id[doc.material_id] = doc
            mp_id_list_by_system[system].append(str(doc.material_id))
        
        print(f"Merging absorption data with mp-base...")
        for system in CRYSTAL_SYSTEMS:
            mp_id_list = mp_id_list_by_system[system]
            abs_docs = mpr.materials.absorption.search(
                material_ids=mp_id_list
            )
            mp_base_json_dict = loadfn(f"data/mp-base/{system}.json")
            
            base_id_set = set(mp_base_json_dict.keys())
            for i in tqdm(range(len(mp_id_list))):
                mp_id = mp_id_list[i]
                if mp_id not in base_id_set:
                    mp_base_json_dict[mp_id] = docs_by_id[mp_id]
                mp_base_json_dict[mp_id]['absorption'] = abs_docs[i].absorption_coefficient

            # save updated system JSON files
            data_raw_dir = f'data/mp-base'
            file_raw_name = f'{system}.json'
            data_raw_path = open_write_file(data_raw_dir, file_raw_name)

            with open(data_raw_path, 'w') as f:
                json.dump(mp_base_json_dict, f, cls=MontyEncoder, indent=4)


def choose_subset_of_mp_data(subset_size=1000) -> None:
    print(f"Subset Size: {subset_size}")
    for system in CRYSTAL_SYSTEMS:
        print(f"Choosing subset of {system} system...")
        mp_raw_json_list = loadfn(f"data/mp-raw/{system}.json")
        mp_subset = random.sample(mp_raw_json_list, k=subset_size)

        # save subset JSON files
        data_raw_dir = f'data/mp-subset'
        file_raw_name = f'{system}.json'
        data_raw_path = open_write_file(data_raw_dir, file_raw_name)

        with open(data_raw_path, 'w') as f:
            json.dump(mp_subset, f, cls=MontyEncoder, indent=4)


def convert_summarydoc_to_cif() -> None:
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

    args = _parse_args()

    clone_mp_raw()
    merge_abs_mp_data()

    # choose_subset_of_mp_data(args.subset_size)
    # convert_summarydoc_to_cif()
