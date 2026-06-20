import argparse
from tqdm import tqdm
import pickle
from utils import CRYSTAL_SYSTEMS, open_write_file


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bounds', action='store_true', help='Computes maximum bond distance and neighbor cnt aross all graphs')
    parser.add_argument('--test', action='store_true', help='Tests all graphs for bidirectionality')
    return parser.parse_args()


def unpack_all_graphs() -> dict:
    graph_dict = dict()
    for system in CRYSTAL_SYSTEMS:
        with open(f'data/graphs-multi/{system}.pkl', 'rb') as file:
            graph_system_dict = pickle.load(file)
        for key, graph in graph_system_dict.items():
            graph_dict[key] = graph.to_undirected()
    return graph_dict


def find_bounds():
    graph_dict = unpack_all_graphs()
    
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


def bid_test():
    '''
    Testing all digraphs to check for bidirectional edges.
    RESULT: No bidirectional edges in all 7000 digraphs.
    '''
    graph_dict = unpack_all_graphs()
    
    bidirectional = False

    print("Testing bidirectionality...")
    for key, value in tqdm(graph_dict.items()):
        graph = value['graph']
        has_bidirectional = any(graph.has_edge(v, u) for u, v in graph.edges() if u != v)
        if has_bidirectional:
            bidirectional = True
            print(key)
    print("Bidirectional edges exist for above digraphs" if bidirectional else "No bidirectional edges in all digraphs")


if __name__ == "__main__":
    args = _parse_args()
    graph_dict = unpack_all_graphs()

    if args.bounds:
        find_bounds()
    if args.test:
        bid_test()