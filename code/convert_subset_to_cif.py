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
CIF_DATA_PATH = 'data/cif'


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset-size', default=1000, type=int, help='size for each class')
    return parser.parse_args()


# not needed
def clone_mp_raw():
    open_write_file('data/mp-base', '')
    for system in CRYSTAL_SYSTEMS:
        shutil.copyfile(f'data/mp-raw/{system}.json', f'data/mp-base/{system}.json')


class CrystalSubset:
    def __init__(self):
        self.fields = FIELDS[3:]

        self.crystals, self.crystal_system = self.load_crystals()

        (
            self.ids_by_system,
            self.avail_by_system,
            self.property_freq,
            self.property_rarity,
        ) = self.compute_prop_distributions_fast()

        self.prop_cnts = np.zeros(len(self.fields), dtype=np.float32)


    def load_crystals(self):
        all_crystals = {}
        crystal_system = {}

        for system in CRYSTAL_SYSTEMS:
            print(f"Loading crystals from {system} system...")
            crystal_json = loadfn(f"data/mp-raw/{system}.json")

            crystal_system[system] = crystal_json
            all_crystals.update(crystal_json)

        return all_crystals, crystal_system


    def compute_prop_distributions_fast(self):
        ids_by_system = {}
        avail_by_system = {}

        property_freq = np.zeros(len(self.fields), dtype=np.float32)

        for system in CRYSTAL_SYSTEMS:
            print(f"Computing property distribution for {system} system...")

            crystal_json = self.crystal_system[system]
            ids = list(crystal_json.keys())

            A = np.zeros((len(ids), len(self.fields)), dtype=np.float32)

            for i, mp_id in enumerate(tqdm(ids)):
                value = crystal_json[mp_id]

                for prop_idx, prop in enumerate(self.fields):
                    x = value[prop]

                    if x is not None:
                        A[i, prop_idx] = 1.0

            ids_by_system[system] = np.asarray(ids, dtype=object)
            avail_by_system[system] = A

            property_freq += A.sum(axis=0)

        property_rarity = 1000.0 / np.sqrt(property_freq + 1.0)

        print("property_freq:", property_freq)
        print("property_rarity:", property_rarity)

        return ids_by_system, avail_by_system, property_freq, property_rarity


    def select_one_system_fast(self, system, subset_size):
        ids = self.ids_by_system[system]
        A = self.avail_by_system[system]

        # Number of available properties for each crystal.
        base_scores = A.sum(axis=1)

        # Initial dynamic weights.
        weights = self.property_rarity / np.sqrt(self.prop_cnts + 1.0)

        # Initial scores for all crystals in this system.
        scores = base_scores + A @ weights

        selected_indices = []

        for _ in tqdm(range(subset_size)):
            # Pick best remaining crystal.
            best_idx = int(np.argmax(scores))
            selected_indices.append(best_idx)

            selected_avail = A[best_idx]

            # Update global selected property counts.
            old_weights = weights
            self.prop_cnts += selected_avail
            weights = self.property_rarity / np.sqrt(self.prop_cnts + 1.0)

            # Only properties present in selected crystal changed counts.
            changed = selected_avail.astype(bool)

            if np.any(changed):
                delta = weights[changed] - old_weights[changed]

                # Incrementally update all scores.
                scores += A[:, changed] @ delta

            # Prevent selecting same crystal again.
            scores[best_idx] = -np.inf

        selected_ids = ids[selected_indices].tolist()
        return selected_ids


    def select_and_save_subset_ids(self, subset_size=6700):
        for system in CRYSTAL_SYSTEMS:
            print(f"Computing optimal subset of {system} system...")
            
            # reset prop counts (each system should be independent)
            self.prop_cnts = np.zeros(len(self.fields), dtype=np.float32)

            subset_id_list = self.select_one_system_fast(system, subset_size)
            subset_json = self.compile_dicts_from_ids(subset_id_list)

            subset_path = open_write_file(SUBSET_DATA_PATH, f"{system}.pkl")
            with open(subset_path, "wb") as f:
                pickle.dump(subset_json, f)


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
            
            for system in CRYSTAL_SYSTEMS:
                print(f"Merging absorption data for {system} system with mp-subset...")
                mp_id_list = mp_id_list_by_system[system]
                abs_docs = mpr.materials.absorption.search(
                    material_ids=mp_id_list
                )
                with open(f"{SUBSET_DATA_PATH}/{system}.pkl", 'rb') as file:
                    subset_json_dict = pickle.load(file)
                
                base_id_set = set(subset_json_dict.keys())
                for i in tqdm(range(len(mp_id_list))):
                    mp_id = mp_id_list[i]
                    if mp_id not in base_id_set:
                        subset_json_dict[mp_id] = docs_by_id[mp_id]
                    subset_json_dict[mp_id]['absorption'] = abs_docs[i].absorption_coefficient

                # save updated subset pickle files
                subset_path = open_write_file(SUBSET_DATA_PATH, f"{system}.pkl")
                with open(subset_path, "wb") as f:
                    pickle.dump(subset_json_dict, f)


def choose_random_subset_of_mp_data(subset_size=1000) -> None:
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


def convert_subsets_to_cif() -> None:
    for system in CRYSTAL_SYSTEMS:
        print(f"Converting {system} system files into CIF...")
        with open(f"{SUBSET_DATA_PATH}/{system}.pkl", 'rb') as file:
            mp_json_list = pickle.load(file)

        data_cif_dir = f'{CIF_DATA_PATH}/{system}'
        for mp_id, value in mp_json_list.items():
            file_cif_name = f'{mp_id}.cif'
            data_cif_path = open_write_file(data_cif_dir, file_cif_name)

            cif_data = value["structure"].to(fmt="cif")
            with open(data_cif_path, 'w') as f:
                f.write(cif_data)


if __name__ == "__main__":
    random.seed(42)

    args = _parse_args()
    crystal_subset = CrystalSubset()
    crystal_subset.select_and_save_subset_ids(subset_size=args.subset_size)
    convert_subsets_to_cif()
