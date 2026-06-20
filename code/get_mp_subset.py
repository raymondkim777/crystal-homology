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

from get_mp_data import save_doc_as_dict
from utils import CRYSTAL_SYSTEMS, FIELDS, open_write_file

from dotenv import load_dotenv
load_dotenv()


MP_DATA_PATH = 'data/mp-raw'
SUBSET_DATA_PATH = 'data/mp-subset'
CIF_DATA_PATH = 'data/cif'


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--random', action='store_true', help='compute random, not optimal subset')
    parser.add_argument('--subset', action='store_true', help='compute subset')
    parser.add_argument('--size', default=6700, type=int, help='subset size for each class')
    parser.add_argument('--cif', action='store_true', help='convert subset to CIF files')
    parser.add_argument('--absorb', action='store_true', help='append absorption data to subsets')
    return parser.parse_args()


class CrystalSubset:
    def __init__(self, subset=False, absorption_data=False):
        self.fields = FIELDS[3:]

        if subset:
            self.crystals, self.crystal_system = self.load_crystals()
            (
                self.ids_by_system,
                self.avail_by_system,
                self.property_freq,
                self.property_rarity,
            ) = self.compute_prop_distributions_fast()

            self.prop_cnts = np.zeros(len(self.fields), dtype=np.float32)

        if absorption_data:
            self.absorption_subset = AbsorptionSubset()


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

        print(f"Computing property distributions for systems...")
        for system in CRYSTAL_SYSTEMS:

            crystal_json = self.crystal_system[system]
            ids = list(crystal_json.keys())

            A = np.zeros((len(ids), len(self.fields)), dtype=np.float32)

            for i, mp_id in tqdm(enumerate(ids), desc=f"{system}: "):
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

        for _ in tqdm(range(subset_size), desc=f"{system}: "):
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
        print(f"Computing optimal subsets...")
        for system in CRYSTAL_SYSTEMS:
            # reset prop counts (each system should be independent)
            self.prop_cnts = np.zeros(len(self.fields), dtype=np.float32)

            subset_id_list = self.select_one_system_fast(system, subset_size)
            subset_json = self.compile_dicts_from_ids(subset_id_list)

            subset_path = open_write_file(SUBSET_DATA_PATH, f"{system}.pkl")
            with open(subset_path, "wb") as f:
                pickle.dump(subset_json, f)


    def compile_dicts_from_ids(self, id_list):
        return {mp_id: self.crystals[mp_id] for mp_id in id_list}
    

    def merge_abs_mp_data(self):
        with MPRester(force_renew=True) as mpr:

            print(f"Querying summary data (absorption)...")
            # list_of_available_fields = mpr.materials.summary.available_fields
            # print(list_of_available_fields)
            docs = mpr.materials.summary.search(
                has_props=[HasProps.absorption],
                fields=FIELDS
            )

            id_list = []
            docs_by_id = dict()
            mp_id_list_by_system = {system: [] for system in CRYSTAL_SYSTEMS}

            for doc in docs:
                id_list.append(doc.material_id)
                docs_by_id[doc.material_id] = doc
                system = str(doc.symmetry.crystal_system).lower()
                mp_id_list_by_system[system].append(str(doc.material_id))

            # query AbsorptionDoc for all ids
            print(f"Collecting absorption data...")
            abs_docs = mpr.materials.absorption.search(
                material_ids=id_list,
            )
            abs_docs_by_id = dict()
            for doc in abs_docs:
                abs_docs_by_id[doc.material_id] = doc
            
            for system in CRYSTAL_SYSTEMS:
                mp_id_list = mp_id_list_by_system[system]
                abs_docs_list = [abs_docs_by_id[id] for id in mp_id_list]

                if len(mp_id_list) == 0:
                    print(f"No absorption data for {system} system!")
                    continue

                print(f"Extracting absorption data for {system} system...")
                # extract absorption features from absorption coefficeints
                # each np.ndarrays, in order of mp_id_list
                (
                    abs_max, 
                    abs_max_e, 
                    abs_int, 
                    abs_int_vis,
                    abs_avg_vis, 
                    abs_onset_e
                ) = self.absorption_subset.extract_absorption_features(abs_docs_list)

                print(f"Merging absorption data with subset...")
                # append to existing subset
                with open(f"{SUBSET_DATA_PATH}/{system}.pkl", 'rb') as file:
                    subset_json_dict = pickle.load(file)
                    
                id_set = set(subset_json_dict.keys())
                for i in tqdm(range(len(mp_id_list)), desc=f'{system}: '):
                    mp_id = mp_id_list[i]
                    if mp_id not in id_set:
                        subset_json_dict[mp_id] = save_doc_as_dict(docs_by_id[mp_id])
                    subset_json_dict[mp_id]['absorption'] = {
                        'max_absorption': abs_max[i], 
                        'max_absorption_energy': abs_max_e[i], 
                        'integrated_absorption': abs_int[i], 
                        'integrated_absorption_visible': abs_int_vis[i], 
                        'average_absorption_visible': abs_avg_vis[i], 
                        'absorption_onset_energy': abs_onset_e[i]
                    }

                # save updated subset pickle files
                subset_path = open_write_file(SUBSET_DATA_PATH, f"{system}.pkl")
                with open(subset_path, "wb") as f:
                    pickle.dump(subset_json_dict, f)

    
    def convert_subsets_to_cif(self) -> None:
        print(f"Converting structure files into CIF...")
        for system in CRYSTAL_SYSTEMS:
            with open(f"{SUBSET_DATA_PATH}/{system}.pkl", 'rb') as file:
                mp_json_list = pickle.load(file)

            data_cif_dir = f'{CIF_DATA_PATH}/{system}'
            for mp_id, value in tqdm(mp_json_list.items(), desc=f"{system}: "):
                file_cif_name = f'{mp_id}.cif'
                data_cif_path = open_write_file(data_cif_dir, file_cif_name)

                cif_data = value["structure"].to(fmt="cif")
                with open(data_cif_path, 'w') as f:
                    f.write(cif_data)


class AbsorptionSubset:
    def __init__(self):
        return


    def find_absorption_onset(
            self, 
            absorption, 
            energy, 
            baseline_fraction=0.03,
            k=2.0,
            min_consecutive=3,
            relative_floor=0.01,
    ):
        '''
        Attempt to find absorption onset energy for one crystal from discrete 
        absorption coefficients and energy levels. 
        
        baseline_fraction: Fraction of lowest-energy points to estimate baseline
        k: Noise multiplier
        min_consecutive: Number of consecutive points above threshold required
        relative_floor: Fraction of max absorption, to estimate min threshold
        '''
        absorption = np.asarray(absorption)
        energy = np.asarray(energy)

        # make sure energy is ascending
        idx = np.argsort(energy)
        energy = energy[idx]
        absorption = absorption[idx]

        # estimate low-energy baseline
        baseline_num = max(3, int(len(energy) * baseline_fraction))
        baseline_region = absorption[:baseline_num]

        baseline = np.median(baseline_region)
        noise = np.std(baseline_region)

        # estimate threshold based on noise/max absorption
        threshold_noise = baseline + k * noise
        threshold_relative = baseline + relative_floor * (np.max(absorption) - baseline)
        threshold = max(threshold_noise, threshold_relative)

        # find absorptions indices above threshold
        above = absorption > threshold

        # Require several consecutive points above threshold
        for i in range(len(above) - min_consecutive + 1):
            if np.all(above[i: i + min_consecutive]):
                # interpolate to find specific energy that crosses threshold
                if i == 0:
                    return energy[i]
                e1, e2 = energy[i - 1], energy[i]
                a1, a2 = absorption[i - 1], absorption[i]

                if a2 == a1:
                    return energy[i]

                return e1 + (threshold - a1) * (e2 - e1) / (a2 - a1)

        raise ValueError("no absorption onset energy computed")
    

    def extract_absorption_features(self, abs_docs):
        all_abs = [doc.absorption_coefficient for doc in abs_docs]
        all_energy = [doc.energies for doc in abs_docs]
        
        all_abs = np.asarray(all_abs)
        all_energy = np.asarray(all_energy)
        assert len(all_abs) == len(all_energy), f'crystal count mismatch!'
        assert all_abs.shape == all_energy.shape, \
            f'absorption/energy data mismatch! {all_abs.shape} vs {all_energy.shape}'
        assert all_abs.ndim == 2, f'invalid absorption data shape of {all_abs.shape}!'
        assert all_energy.ndim == 2, f'invalid energy data shape of {all_energy.shape}!!'

        # maximum absorption & corresponding energy
        abs_max_idx = np.argmax(all_abs, axis=1)
        abs_max = all_abs[np.arange(all_abs.shape[0]), abs_max_idx]
        abs_max_e = all_energy[np.arange(all_energy.shape[0]), abs_max_idx]

        # integrated absorption
        abs_int = []
        for i in range(len(all_abs)):
            # make sure energies are ascending
            idx = np.argsort(all_energy[i])
            abs_sorted = all_abs[i][idx]
            e_sorted = all_energy[i][idx]

            # compute integral
            area = np.trapezoid(abs_sorted, e_sorted)
            abs_int.append(area)
        abs_int = np.asarray(abs_int)

        # (visible range) integrated absorption & average absorption
        abs_avg_vis = []
        abs_int_vis = []
        for i in range(len(all_abs)):
            # energies within visible light range
            mask = (all_energy[i] >= 1.6) & (all_energy[i] <= 3.3)
            abs_sub = all_abs[i][mask]
            e_sub = all_energy[i][mask]

            # average absorption
            abs_avg_vis.append(np.mean(abs_sub))

            # make sure energies are ascending
            idx = np.argsort(e_sub)
            abs_sorted = abs_sub[idx]
            e_sorted = e_sub[idx]

            # compute integral
            area = np.trapezoid(abs_sorted, e_sorted)
            abs_int_vis.append(area)
        abs_avg_vis = np.asarray(abs_avg_vis)
        abs_int_vis = np.asarray(abs_int_vis)

        # absorption onset energy
        abs_onset_e = np.asarray([
            self.find_absorption_onset(all_abs[i], all_energy[i])
            for i in range(len(all_abs))
        ])

        return abs_max, abs_max_e, abs_int, abs_int_vis, abs_avg_vis, abs_onset_e


##### DEPRECATED #####
    
def select_random_subset(subset_size=6700) -> None:
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


if __name__ == "__main__":
    random.seed(42)

    args = _parse_args()
    crystal_subset = CrystalSubset(
        subset=args.subset,
        absorption_data=args.absorb
    )

    if args.random:
        select_random_subset()
        if args.cif:
            crystal_subset.convert_subsets_to_cif()
    else:
        if args.subset:
            crystal_subset.select_and_save_subset_ids(subset_size=args.size)
        if args.absorb:
            crystal_subset.merge_abs_mp_data()
        if args.cif:
            crystal_subset.convert_subsets_to_cif()
