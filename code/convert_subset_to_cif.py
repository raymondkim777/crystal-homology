import argparse
import shutil
import json
import random
import pickle
import numpy as np
from tqdm import tqdm
from mp_api.client import MPRester
from monty.json import MontyEncoder
from monty.serialization import loadfn
from emmet.core.summary import HasProps
from utils import CRYSTAL_SYSTEMS, FIELDS, open_write_file

from dotenv import load_dotenv
load_dotenv()


MP_DATA_PATH = 'data/mp-raw'
SUBSET_DATA_PATH = 'data/mp-subset'


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset-size', default=1000, type=int, help='size for each class')
    return parser.parse_args()


def clone_mp_raw():
    open_write_file('data/mp-base', '')
    for system in CRYSTAL_SYSTEMS:
        shutil.copyfile(f'data/mp-raw/{system}.json', f'data/mp-base/{system}.json')


class CrystalSubset():
    def __init__(self):
        self.fields = FIELDS[3:]
        self.crystals, self.crystal_system = self.load_crystals()
        prop_dist = self.compute_prop_distributions() 
        self.crystal_avail, self.property_freq, self.property_rarity = prop_dist
        self.prop_cnts = np.asarray([0 for _ in self.fields])

        # [26609, 13028, 62972, 79081, 81917, 88185, 88185, 154305, 22560]
        # [6.130241202150422, 8.76081395948087, 3.984949373681705, 3.555995296929075, 3.4938988652363726, 3.3674424164483, 3.3674424164483, 2.5457080348553728, 6.657647963521928]
    
    
    def load_crystals(self):
        all_crystals = dict()
        crystal_system = dict()
        for system in CRYSTAL_SYSTEMS:
            print(f"Loading crystals from {system} system...")
            crystal_json = loadfn(f"data/mp-base/{system}.json")
            crystal_system[system] = crystal_json
            all_crystals = all_crystals | crystal_json
        return all_crystals, crystal_system
    

    def compute_prop_distributions(self):
        crystal_avail = dict()  # contains np arrays
        property_freq = [0 for i in self.fields]

        for system in CRYSTAL_SYSTEMS:
            print(f"Computing property distribution for {system} system...")
            crystal_json = loadfn(f"data/mp-base/{system}.json")
            for mp_id, value in tqdm(crystal_json.items()):
                avail = []
                for prop_idx in range(len(self.fields)):
                    prop = self.fields[prop_idx]
                    if value[prop] is None or value[prop] == 0.0:
                        avail.append(0)
                    else:
                        avail.append(1)
                        property_freq[prop_idx] += 1
                crystal_avail[mp_id] = np.asarray(avail)
        property_rarity = [1000 / np.sqrt(freq + 1) for freq in property_freq]

        print(property_freq)
        print(property_rarity)
        return crystal_avail, property_freq, property_rarity


    def compute_crystal_score(self, mp_id, prop_cnts):
        score = 0
        for prop_idx in range(len(self.fields)):
            prop_rare_w = self.crystal_avail[mp_id][prop_idx]
            prop_rare_w *=  self.property_rarity[prop_idx]
            prop_rare_w *= 1 / np.sqrt(prop_cnts[prop_idx] + 1)

            prop_cnt_w = 0.2 * np.sum(self.crystal_avail[mp_id])
            score += prop_rare_w + prop_cnt_w
        return score


    def select_subset_ids(self, subset_size=1000):
        subset_ids = dict()
        for system in CRYSTAL_SYSTEMS:
            print(f"Computing optimal subset of {system} system...")
            system_subset_ids = []
            remaining_ids = set(self.crystal_system[system].keys())
            for _ in tqdm(range(subset_size)):
                id_score_pairs = [
                    (mp_id, self.compute_crystal_score(mp_id, self.prop_cnts)) 
                    for mp_id in remaining_ids
                ]
                id_score_pairs.sort(reverse=True, key=lambda x: x[1])
                selected_id = id_score_pairs[0][0]
                system_subset_ids.append(selected_id)
                remaining_ids.remove(selected_id)
                self.prop_cnts += self.crystal_avail[selected_id]
            subset_ids[system] = system_subset_ids

        # save all subset_ids JSON
        subset_jsons = self.compile_dicts_from_ids(subset_ids)
        subset_path = open_write_file(SUBSET_DATA_PATH, f'{system}.pkl')
        with open(subset_path, 'wb') as f:
            pickle.dump(subset_jsons, f)


    def compile_dicts_from_ids(self, id_list):
        return {mp_id: self.crystals[mp_id] for mp_id in id_list}
                




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
    
    crystal_subset = CrystalSubset()
    crystal_subset.select_subset_ids(subset_size=args.subset_size)

    # compute_prop_distributions()
    # merge_abs_mp_data()

    # choose_subset_of_mp_data(args.subset_size)
    # convert_summarydoc_to_cif()
