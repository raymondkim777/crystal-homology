import argparse
import pickle
import json
import csv
import numpy as np

from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import CRYSTAL_SYSTEMS, FIELDS, PREDICT, ABS_PREDICT, TASK_SPECS, ABS_TASK_SPECS, open_write_file


# DATA_DIRECTORY = "data/pretrain"
# DATA_DIRECTORY = "data/abs"
# DATA_SUBSET_PATH = f"{DATA_DIRECTORY}/mp-subset"
# DATA_SUBSET_PATH = f"{DATA_DIRECTORY}/mp-abs"
DATA_DIRECTORY = None
DATA_SUBSET_PATH = None
MULTIGRAPH_DIRECTORY = None


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--abs', action='store_true', help='Only focuses separately on data/abs')
    parser.add_argument('--merge', action='store_true', help='Pretrain data contains absorption data')
    parser.add_argument('--bounds', action='store_true', help='Computes maximum bond distance and neighbor cnt aross all graphs')
    parser.add_argument('--avail', action='store_true', help='Computes availability for each property in dataset')
    parser.add_argument('--dist', action='store_true', help='Computes label distribution for each property in dataset')
    parser.add_argument('--test', action='store_true', help='Tests all graphs for bidirectionality')
    return parser.parse_args()


def unpack_all_graphs(undirected=False) -> dict:
    graph_dict = dict()
    for system in CRYSTAL_SYSTEMS:
        print(f"Unpacking multigraphs from {system} system...")
        with open(f'{MULTIGRAPH_DIRECTORY}/{system}.pkl', 'rb') as file:
            graph_system_dict = pickle.load(file)
        for key, graph in graph_system_dict.items():
            graph_dict[key] = graph.to_undirected() if undirected else graph
    return graph_dict


def find_graph_bounds():
    # pretrain/abs is already defined
    graph_dict = unpack_all_graphs(undirected=False)

    # computing bounds
    max_num_nbr = 0
    max_bond_dist = 0

    print("Computing bounds...")
    for key, graph in tqdm(graph_dict.items()):
        # max_degree = max(d for _, d in graph.out_degree())  # for digraphss
        max_degree = max(d for _, d in graph.degree())
        max_num_nbr = max(max_num_nbr, max_degree)

        weights = [0] + [data['weight'] for _, _, data in graph.edges(data=True)]
        if len(weights) == 0:
            print(f"No edges: {key}")
            pass
        max_dist = max(weights)
        max_bond_dist = max(max_bond_dist, max_dist)

    print("Maximum Neighbor Cnt:", max_num_nbr)
    print("Maximum Bond Distance:", max_bond_dist)

    # save values
    val_json = {
        'max_num_nbr': max_num_nbr, 
        'max_bond_dist': max_bond_dist,
    }
    json_path = open_write_file(DATA_DIRECTORY, 'bounds.json')
    with open(json_path, 'w') as f:
        json.dump(val_json, f, indent=4)


def find_data_avail():
    fields = FIELDS[3:] if not args.abs else ABS_PREDICT
    all_crystals = {}
    crystal_system = {}

    for system in CRYSTAL_SYSTEMS:
        print(f"Loading crystals from {system} system...")
        with open(f'{DATA_SUBSET_PATH}/{system}.pkl', 'rb') as file:
            crystal_dict = pickle.load(file)
        crystal_system[system] = crystal_dict
        all_crystals.update(crystal_dict)

    property_ratios = dict()

    print(f"Computing property distributions for systems...")
    for system in CRYSTAL_SYSTEMS:

        crystal_json = crystal_system[system]
        ids = list(crystal_json.keys())

        A = np.zeros((len(ids), len(fields)), dtype=np.float32)

        for i, mp_id in tqdm(enumerate(ids), desc=f"{system}: "):
            value = crystal_json[mp_id]

            for prop_idx, prop in enumerate(fields):
                x = value[prop]

                if x is not None:
                    A[i, prop_idx] = 1.0

        property_ratios[system] = A.sum(axis=0) / len(ids)
    
    csv_data = [['system'] + [field for field in fields]] + [
        [system] + list(property_ratios[system]) for system in CRYSTAL_SYSTEMS
    ]
    csv_path = open_write_file(f'{DATA_DIRECTORY}/stats', 'avail.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)


def find_data_dist(abs, merge):
    # collect all crystal docs
    all_crystal_list = []
    for system in CRYSTAL_SYSTEMS:
        print(f"Loading crystals from {system} system...")
        with open(f'{DATA_SUBSET_PATH}/{system}.pkl', 'rb') as file:
            crystal_dict = pickle.load(file)
        all_crystal_list += list(crystal_dict.values())
    
    props = ABS_PREDICT if abs else PREDICT
    tasks = ABS_TASK_SPECS if abs else TASK_SPECS
    all_values = {prop: [] for prop in props}

    # store all dataset values
    system_to_int = {CRYSTAL_SYSTEMS[idx]: idx for idx in range(len(CRYSTAL_SYSTEMS))}
    for doc in tqdm(all_crystal_list):    
        if not abs:
            all_values['system'].append(system_to_int[str(doc['symmetry'].crystal_system).lower()])
            if doc['bandstructure'] is not None and doc['bandstructure'].latimer_munro is not None:
                all_values['direct_gap'].append(doc['bandstructure'].latimer_munro.direct_gap)
            if doc['band_gap'] is not None:
                all_values['band_gap'].append(doc['band_gap'])
            if doc['efermi'] is not None:
                all_values['efermi'].append(doc['efermi'])
            if doc['is_gap_direct'] is not None:
                all_values['is_gap_direct'].append(int(doc['is_gap_direct']))
            
            if merge and 'absorption' in doc.keys():
                doc_abs = doc['absorption']
                if doc_abs['max_absorption'] is not None:
                    all_values['max_absorption'].append(doc_abs['max_absorption'])
                if doc_abs['max_absorption_energy'] is not None:
                    all_values['max_absorption_energy'].append(doc_abs['max_absorption_energy'])
                if doc_abs['integrated_absorption'] is not None:
                    all_values['integrated_absorption'].append(doc_abs['integrated_absorption'])
                if doc_abs['integrated_absorption_visible'] is not None:
                    all_values['integrated_absorption_visible'].append(doc_abs['integrated_absorption_visible'])
                if doc_abs['average_absorption_visible'] is not None:
                    all_values['average_absorption_visible'].append(doc_abs['average_absorption_visible'])
                if doc_abs['absorption_onset_energy'] is not None:
                    all_values['absorption_onset_energy'].append(doc_abs['absorption_onset_energy'])
        else:
            doc_abs = doc
            if doc_abs['max_absorption'] is not None:
                all_values['max_absorption'].append(doc_abs['max_absorption'])
            if doc_abs['max_absorption_energy'] is not None:
                all_values['max_absorption_energy'].append(doc_abs['max_absorption_energy'])
            if doc_abs['integrated_absorption'] is not None:
                all_values['integrated_absorption'].append(doc_abs['integrated_absorption'])
            if doc_abs['integrated_absorption_visible'] is not None:
                all_values['integrated_absorption_visible'].append(doc_abs['integrated_absorption_visible'])
            if doc_abs['average_absorption_visible'] is not None:
                all_values['average_absorption_visible'].append(doc_abs['average_absorption_visible'])
            if doc_abs['absorption_onset_energy'] is not None:
                all_values['absorption_onset_energy'].append(doc_abs['absorption_onset_energy'])

    open_write_file(f'{DATA_DIRECTORY}/stats', '')
    for prop, values in all_values.items():
        if tasks[prop]['head'] in ['binary', 'multiclass']:
            labels, counts = np.unique(values, return_counts=True)
            csv_data = [['Label', 'Counts']]
            csv_data += [[labels[i], counts[i]] for i in range(len(labels))]
            csv_path = open_write_file(f'{DATA_DIRECTORY}/stats', f'{prop}.csv')
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(csv_data)

            plt.clf()
            plt.bar(labels, counts, color='skyblue', edgecolor='black')
            plt.title(f'{prop} label distribution')
            plt.xlabel(f'{prop} classes')
            plt.ylabel('frequency')
            plt.savefig(f"{DATA_DIRECTORY}/stats/{prop}.png")

        elif tasks[prop]['head'] == 'regression':
            plt.clf()
            counts, bin_edges, patches = plt.hist(values, bins=30, edgecolor='black', color='skyblue')
            csv_data = [['Intervals', 'Counts']]
            csv_data += [[f'{bin_edges[i]:.2f}~{bin_edges[i + 1]:.2f}', counts[i]] for i in range(len(counts))]
            csv_path = open_write_file(f'{DATA_DIRECTORY}/stats', f'{prop}.csv')
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(csv_data)
            plt.title(f"{prop} label distribution")
            plt.xlabel(f"{prop} value intervals")
            plt.ylabel("frequency")
            plt.savefig(f"{DATA_DIRECTORY}/stats/{prop}.png")
        else:
            raise ValueError(f"[Find Dist] Unrecognized task {tasks[prop]}")
        pass

     


def bid_test():
    '''
    Testing all digraphs to check for bidirectional edges.
    '''
    graph_dict = unpack_all_graphs(undirected=False)
    bidirectional = False

    bid_id_list = []
    json_path = open_write_file(DATA_DIRECTORY, 'bid_true.txt')

    print("Testing bidirectionality...")
    for key, graph in tqdm(graph_dict.items()):
        has_bidirectional = any(
            graph.has_edge(v, u) 
            for u, v in graph.edges() if u != v
        )
        if has_bidirectional:
            bidirectional = True
            bid_id_list.append(key)

    if len(bid_id_list) != 0:
        with open(json_path, 'w') as f:
            for id in bid_id_list:
                f.write(f"{id}\n")

    print(f"Bidirectional edges exist for digraphs, check {json_path}" 
          if bidirectional else "No bidirectional edges in all digraphs")


if __name__ == "__main__":
    args = _parse_args()

    DATA_DIRECTORY = "data/abs" if args.abs else "data/pretrain"
    DATA_SUBSET_PATH = f"{DATA_DIRECTORY}/mp-abs" if args.abs else f"{DATA_DIRECTORY}/mp-subset"
    MULTIGRAPH_DIRECTORY = f"{DATA_DIRECTORY}/graphs-multi"

    assert not (args.abs and args.merge), "--abs changes directory to data/abs --> can't also do --merge"

    if args.bounds:
        find_graph_bounds()
    if args.avail:
        find_data_avail()
    if args.dist:
        find_data_dist(abs=args.abs, merge=args.merge)
    if args.test:
        bid_test()