import argparse
import pickle
import json
import csv
import numpy as np

from tqdm import tqdm
from utils import CRYSTAL_SYSTEMS, FIELDS, DIMENSION_CNT, open_write_file


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bounds', action='store_true', help='Computes maximum bond distance and neighbor cnt aross all graphs')
    parser.add_argument('--stats', action='store_true', help='Computes mean/stdev for each scalar prediction value')
    parser.add_argument('--test', action='store_true', help='Tests all graphs for bidirectionality')
    return parser.parse_args()


def unpack_all_graphs(undirected=False) -> dict:
    graph_dict = dict()
    for system in CRYSTAL_SYSTEMS:
        print(f"Unpacking multigraphs from {system} system...")
        with open(f'data/graphs-multi/{system}.pkl', 'rb') as file:
            graph_system_dict = pickle.load(file)
        for key, graph in graph_system_dict.items():
            graph_dict[key] = graph.to_undirected() if undirected else graph
    return graph_dict


def find_graph_bounds():
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
    json_path = open_write_file('data/', 'bounds.json')
    with open(json_path, 'w') as f:
        json.dump(val_json, f, indent=4)


def find_pred_stats():
    fields = FIELDS[3:]
    all_crystals = {}
    crystal_system = {}

    for system in CRYSTAL_SYSTEMS:
        print(f"Loading crystals from {system} system...")
        with open(f'data/mp-subset/{system}.pkl', 'rb') as file:
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
    csv_path = open_write_file('data', 'stats.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)


def bid_test():
    '''
    Testing all digraphs to check for bidirectional edges.
    '''
    graph_dict = unpack_all_graphs(undirected=False)
    bidirectional = False

    bid_id_list = []
    json_path = open_write_file('data/', 'bid_true.txt')

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

    if args.bounds:
        find_graph_bounds()
    if args.stats:
        find_pred_stats()
    if args.test:
        bid_test()