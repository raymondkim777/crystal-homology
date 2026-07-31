import argparse
import os
import shutil
import json
import csv
import pickle
import random
import math
import numpy as np
from tqdm import tqdm
from mp_api.client import MPRester
from monty.json import MontyEncoder
from monty.serialization import loadfn
from emmet.core.summary import HasProps

from pymatgen.io.cif import CifParser
from pymatgen.core import Structure
from pymatgen.core.composition import Composition
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, SymmetryUndeterminedError

from get_mp_data import save_doc_as_dict
from utils import CRYSTAL_SYSTEMS, FIELDS, open_write_file

from dotenv import load_dotenv
load_dotenv()


DATA_PATH = 'data'
MP_DATA_PATH = f'{DATA_PATH}/mp-raw'

DATA_PRE_PATH = f'{DATA_PATH}/pretrain'
DATA_PRE_SUBSET_PATH = f'{DATA_PRE_PATH}/mp-subset'
DATA_PRE_CIF_PATH = f'{DATA_PRE_PATH}/cif'

DATA_ABS_PATH = f'{DATA_PATH}/abs'
DATA_ABS_SUBSET_PATH = f'{DATA_ABS_PATH}/mp-abs'
DATA_ABS_CIF_PATH = f'{DATA_ABS_PATH}/cif'

DATA_PLQY_PATH = f'{DATA_PATH}/plqy'
DATA_PLQY_SUBSET_PATH = f'{DATA_PLQY_PATH}/mp-plqy'
DATA_PLQY_CIF_PATH_RAW = f'{DATA_PLQY_PATH}/cif-plqy'
DATA_PLQY_CIF_PATH = f'{DATA_PLQY_PATH}/cif'

PROBLEM_CIFS = ['2349961.cif']


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', action='store_true', help='sets random seed to 42')
    parser.add_argument('--subset', action='store_true', help='compute subset')
    parser.add_argument('--size', default=5500, type=int, help='subset size for each class')
    parser.add_argument('--large', action='store_true', help='add random subset of larger non-optimal crystals')
    parser.add_argument('--size-large', default=7000, type=int, help='total subset size (including large) for each class')
    parser.add_argument('--cif', action='store_true', help='convert subset to CIF files')
    parser.add_argument('--abs', action='store_true', help='collect absorption data')
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
            self.abs_id_set = set()


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
                        assert self.crystals[mp_id][prop] is not None, "full crystals and system dict don't match!"
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

        # Do not pick crystals from abs.
        for i in range(len(scores)):
            if ids[i] in self.abs_id_set:   # if no abs data, then set is empty
                scores[i] = -np.inf

        for _ in tqdm(range(subset_size), desc=f"{system}: "):
            # Pick best remaining crystal.
            best_idx = int(np.argmax(scores))

            # add crystal to list
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


    def select_and_save_subset_ids(
            self, subset_size=5500, 
            subset_large=False, total_size=7000, 
    ):  
        print(f"Computing optimal subsets...")
        for system in CRYSTAL_SYSTEMS:
            # reset prop counts (each system should be independent)
            self.prop_cnts = np.zeros(len(self.fields), dtype=np.float32)

            # find optimal subset IDs
            subset_id_list = self.select_one_system_fast(system, subset_size)

            # optionally supplement with larger crystals for generalization
            if subset_large:
                print(f"Supplementing dataset to class size {total_size}...")
                num_add = total_size - len(subset_id_list)
                remaining_ids = tuple(set(self.crystal_system[system].keys()) - set(subset_id_list))
                subset_add_list = random.sample(remaining_ids, k=num_add)
                subset_id_list += subset_add_list

            # compile crystal data as dictionary
            subset_json = self.compile_dicts_from_ids(subset_id_list)

            # save dictionary as pickle
            subset_path = open_write_file(DATA_PRE_SUBSET_PATH, f"{system}.pkl")
            with open(subset_path, "wb") as f:
                pickle.dump(subset_json, f)


    def compile_dicts_from_ids(self, id_list):
        return {mp_id: self.crystals[mp_id] for mp_id in id_list}
    

    def collect_abs_mp_data(self):
        with MPRester(force_renew=True) as mpr:

            print(f"Querying summary data (absorption)...")
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
                    subset_json_dict = dict()
                    subset_abs_path = open_write_file(DATA_ABS_SUBSET_PATH, f'{system}.pkl')
                    with open(subset_abs_path, 'wb') as f:
                        pickle.dump(subset_json_dict, f)
                    continue
            
                self.abs_id_set = set(id_list)

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

                print(f"Saving absorption data...")
                subset_json_dict = dict()

                for i in tqdm(range(len(mp_id_list)), desc=f'{system}: '):
                    mp_id = mp_id_list[i]
                    subset_json_dict[mp_id] = {
                        'structure': docs_by_id[mp_id]['structure'],
                        'max_absorption': abs_max[i], 
                        'max_absorption_energy': abs_max_e[i], 
                        'integrated_absorption': abs_int[i], 
                        'integrated_absorption_visible': abs_int_vis[i], 
                        'average_absorption_visible': abs_avg_vis[i], 
                        'absorption_onset_energy': abs_onset_e[i]
                    }

                # save abs pickle file 
                subset_abs_path = open_write_file(DATA_ABS_SUBSET_PATH, f'{system}.pkl')
                with open(subset_abs_path, 'wb') as f:
                    pickle.dump(subset_json_dict, f)
    

    def convert_subsets_to_cif(self, absorb=False) -> None:
        print(f"Converting structure files into CIF...")
        for system in CRYSTAL_SYSTEMS:
            with open(f"{DATA_PRE_SUBSET_PATH}/{system}.pkl", 'rb') as file:
                mp_json_list = pickle.load(file)

            data_cif_dir = f'{DATA_PRE_CIF_PATH}/{system}'
            for mp_id, value in tqdm(mp_json_list.items(), desc=f"{system}: "):
                file_cif_name = f'{mp_id}.cif'
                data_cif_path = open_write_file(data_cif_dir, file_cif_name)

                cif_data = value["structure"].to(fmt="cif")
                with open(data_cif_path, 'w') as f:
                    f.write(cif_data)
        shutil.make_archive(f'{DATA_PRE_PATH}/cif-pre', 'zip', DATA_PRE_CIF_PATH)
        
        if absorb:
            print(f"Converting abs structure files into CIF...")
            for system in CRYSTAL_SYSTEMS:
                with open(f"{DATA_ABS_SUBSET_PATH}/{system}.pkl", 'rb') as file:
                    mp_json_list = pickle.load(file)

                data_cif_dir = f'{DATA_ABS_CIF_PATH}/{system}'
                open_write_file(data_cif_dir, '')
                for mp_id, value in tqdm(mp_json_list.items(), desc=f"{system}: "):
                    file_cif_name = f'{mp_id}.cif'
                    data_cif_path = open_write_file(data_cif_dir, file_cif_name)

                    cif_data = value["structure"].to(fmt="cif")
                    with open(data_cif_path, 'w') as f:
                        f.write(cif_data)
            shutil.make_archive(f'{DATA_ABS_PATH}/cif-abs', 'zip', DATA_ABS_CIF_PATH)


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

        # ! check if abs_onset_e is NaN or Inf
        if np.isnan(abs_onset_e).any():
            coords = np.argwhere(np.isnan(abs_onset_e))
            for coord in coords:
                print(f"NaN value for {abs_docs[coord[0]].material_id}")
        if np.isinf(abs_onset_e).any():
            coords = np.argwhere(np.isinf(abs_onset_e))
            for coord in coords:
                print(f"Inf value for {abs_docs[coord[0]].material_id}")

        return abs_max, abs_max_e, abs_int, abs_int_vis, abs_avg_vis, abs_onset_e

    
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
    args = _parse_args()

    if args.seed:
        random.seed(42)

    crystal_subset = CrystalSubset(
        subset=args.subset,
        absorption_data=args.abs
    )

    if args.abs:
        crystal_subset.collect_abs_mp_data()
    if args.subset:
        crystal_subset.select_and_save_subset_ids(
            subset_size=args.size,
            subset_large=args.large,
            total_size=args.size_large,
        )
    if args.cif:
        crystal_subset.convert_subsets_to_cif(absorb=args.abs)