from tqdm import tqdm
import random
import argparse
import numpy as np
import networkx as nx
from tqdm import tqdm
import pickle
from utils import CRYSTAL_SYSTEMS, open_write_file, plot_nxgraph


CGCNN_DATAPATH = 'cgcnn/data/graph_data'


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', action='store_true', help='Saves graph data to CGCNN data path')
    return parser.parse_args()


def unpack_all_graphs() -> dict:
    graph_dict = dict()
    for system in CRYSTAL_SYSTEMS:
        print(f"Unpacking {system} system...")
        with open(f'data/graphs-multi/{system}.pkl', 'rb') as file:
            graph_dict[system] = pickle.load(file)
    return graph_dict


def graph_process(save=False) -> dict:
    """
    Saves all graphs with labeled crystal systems in CGCNN data folder. 
    Computes and prints required bounds for GraphData.
    Graph structure:
    mp_id: {
        graph: <graph>, 
        system: <system>,
        ... (additional properties to be added)
    }
    """
    graph_dict = unpack_all_graphs()
    new_graph_dict = dict()  # mp_id: {graph: <graph>, system: 'system'}

    # UNDIRECTED edges
    for system in CRYSTAL_SYSTEMS:
        for key, graph in graph_dict[system].items():
            new_graph_dict[key] = {
                'graph': graph.to_undirected(),     # UNDIRECTED (for message passing)
                'system': system
            }
    
    # computing bounds
    max_num_nbr = 0
    max_bond_dist = 0

    print("Computing bounds...")
    for key, value in tqdm(new_graph_dict.items()):
        graph = value['graph']
        # max_degree = max(d for _, d in graph.out_degree())  # for digraphss
        max_degree = max(d for _, d in graph.degree())
        max_num_nbr = max(max_num_nbr, max_degree)

        weights = [0] + [data['weight'] for _, _, data in graph.edges(data=True)]
        if len(weights) == 0:
            # print(key)
            pass
        max_dist = max(weights)
        max_bond_dist = max(max_bond_dist, max_dist)

    print("Maximum Neighbor Cnt:", max_num_nbr)
    print("Maximum Bond Distance:", max_bond_dist)

    if save:
        print(f"Saving files to {CGCNN_DATAPATH}")
        # save graphs to CGCNN data folder
        for mp_id, value in new_graph_dict.items():
            cgcnn_datapath = open_write_file(f"{CGCNN_DATAPATH}/graphs", f'{mp_id}.pkl')
            with open(cgcnn_datapath, 'wb') as f:
                pickle.dump(value, f)
        
        # create id_prop.csv
        system_to_int = {CRYSTAL_SYSTEMS[idx]: idx for idx in range(len(CRYSTAL_SYSTEMS))}
        
        csv_filepath = open_write_file(CGCNN_DATAPATH, 'id_prop.csv')
        with open(csv_filepath, 'w') as f:
            # CRYSTAL SYSTEM - classification
            for mp_id, value in new_graph_dict.items():
                f.write(f"{mp_id[3:]}, {system_to_int[value['system']]}\n")


def bid_test():
    '''
    Testing all digraphs to check for bidirectional edges.
    RESULT: No bidirectional edges in all 7000 digraphs.
    '''
    graph_dict = unpack_all_graphs()
    new_graph_dict = dict()  # mp_id: {graph: <graph>, system: 'system'}
    for system in CRYSTAL_SYSTEMS:
        for key, value in graph_dict[system].items():
            new_graph_dict[key] = {
                'graph': value, 
                'system': system
            }
    
    no_bidirectional = True
    for key, value in new_graph_dict.items():
        graph = value['graph']
        has_bidirectional = any(graph.has_edge(v, u) for u, v in graph.edges() if u != v)
        if has_bidirectional:
            no_bidirectional = False
            print(key)
    print(no_bidirectional)


def test():
    graph_dict = unpack_all_graphs()
    new_graph_dict = dict()  # mp_id: {graph: <graph>, system: 'system'}
    for system in CRYSTAL_SYSTEMS:
        for key, value in graph_dict[system].items():
            new_graph_dict[key] = {
                'graph': value, 
                'system': system
            }
    
    random.seed(42)
    sample = random.sample(list(new_graph_dict.keys()), 20)
    print(sample)


if __name__ == "__main__":
    args = _parse_args()
    graph_process(args.save)
    # bid_test()
    # test()