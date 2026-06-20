import random
import argparse
import numpy as np
import networkx as nx
from tqdm import tqdm
import pickle
import warnings
from utils import CRYSTAL_SYSTEMS, open_write_file, plot_nxgraph
from create_bonds import get_structures_from_cif


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
            graph_system_dict = pickle.load(file)
        for key, graph in graph_system_dict.items():
            structure_filename = f'data/cif/{system}/{key}.cif'
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                structure = get_structures_from_cif(structure_filename)[0]
            graph_dict[key] = {
                'graph': graph.to_undirected(),     # UNDIRECTED (for message passing)
                'system': system, 
                'lattice_matrix': structure.lattice.matrix,
            }
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

    if save:
        print(f"Saving files to {CGCNN_DATAPATH}")
        # save graphs to CGCNN data folder
        for mp_id, value in graph_dict.items():
            cgcnn_datapath = open_write_file(f"{CGCNN_DATAPATH}/graphs", f'{mp_id}.pkl')
            with open(cgcnn_datapath, 'wb') as f:
                pickle.dump(value, f)
        
        # create id_prop.csv
        system_to_int = {CRYSTAL_SYSTEMS[idx]: idx for idx in range(len(CRYSTAL_SYSTEMS))}
        
        csv_filepath = open_write_file(CGCNN_DATAPATH, 'id_prop.csv')
        with open(csv_filepath, 'w') as f:
            # CRYSTAL SYSTEM - classification
            for mp_id, value in graph_dict.items():
                f.write(f"{mp_id[3:]}, {system_to_int[value['system']]}\n")


if __name__ == "__main__":
    args = _parse_args()
    graph_process(args.save)