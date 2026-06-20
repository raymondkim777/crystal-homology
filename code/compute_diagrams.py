import os
import json
import numpy as np
import networkx as nx
from tqdm import tqdm
import pickle
from utils import CRYSTAL_SYSTEMS, DIMENSION_CNT, get_num_cpus, open_write_file, get_max_dist
from gtda.homology import FlagserPersistence
from gtda.plotting import plot_diagram
    

GRAPH_DIRECTORY = "data/graphs"
DIAGRAM_DIRECTORY = "data/diagrams"
MAX_DIST = get_max_dist()


def unpack_all_graphs() -> dict:
    graph_dict = dict()
    for system in CRYSTAL_SYSTEMS:
        with open(f'data/graphs/{system}.pkl', 'rb') as file:
            graph_dict[system] = pickle.load(file)
    return graph_dict


def convert_graph_to_adj_mat(graph: nx.DiGraph) -> np.ndarray:
    
    adj_mat = nx.to_numpy_array(graph, weight='weight', nonedge=np.inf)
    # set diagonals to 0 (just in case)
    np.fill_diagonal(adj_mat, 0)
    return adj_mat


def plot_persistence_diagram(mp_id, diagram) -> None:
    fig = plot_diagram(diagram)
    # fig.show()
    fig.save(f"{DIAGRAM_DIRECTORY}/diagram_{mp_id}.png")


def compute_persistence_diagrams(dims: tuple=tuple(range(DIMENSION_CNT))) -> None:
    n_jobs = get_num_cpus()
    print(f"Using {n_jobs} job processes")

    print(f"Unpacking all graphs...")
    graph_dict = unpack_all_graphs()

    flagser = FlagserPersistence(
        homology_dimensions=dims,
        directed=True,
        filtration='max', 
        coeff=2, 
        max_edge_weight=MAX_DIST,
        infinity_values=None, 
        n_jobs=n_jobs   # parallel processing
    )

    # fit flagser to all graphs
    adj_mat_dict = dict()  # all graphs
    adj_mat_system = dict()  # all graphs divided by system
    for system in CRYSTAL_SYSTEMS:
        adj_mat_system[system] = dict()
        for mat_id, graph in graph_dict[system].items():
            adj_mat = convert_graph_to_adj_mat(graph) 
            adj_mat_system[system][mat_id] = adj_mat
            adj_mat_dict[mat_id] = adj_mat

    adj_mat_list = list(adj_mat_dict.values())
    flagser.fit(adj_mat_list)

    # compute diagrams for each system
    for system in CRYSTAL_SYSTEMS:
        print(f"Computing PD for {system}...")
        
        adj_mat_list = list(adj_mat_system[system].values())
        diagrams = flagser.transform(adj_mat_list)

        # match diagrams to mat_id
        keys = list(adj_mat_system[system].keys())
        diagrams_with_id = {
            keys[i] : diagrams[i]
            for i in range(len(diagrams))
        }

        # print diagram info
        print(f"diagram cnt: {len(diagrams)}")
        graph_diagram = diagrams[0]
        print("Persistence Diagram Shape:", graph_diagram.shape)
        print("Points (Birth, Death, Homology Dimension):\n", graph_diagram)
    
        # save diagrams dict as pickle
        diag_filepath = open_write_file(DIAGRAM_DIRECTORY, f'{system}.pkl')
        with open(diag_filepath, 'wb') as f:
            pickle.dump(diagrams_with_id, f)


def test():
    with open('data/diagrams/cubic.pkl', 'rb') as file:
        diagrams = pickle.load(file)
    
    index = 100
    keys_list = list(diagrams.keys())
    print(keys_list[index])
    plot_persistence_diagram(keys_list[index], diagrams[keys_list[index]])
    pass


if __name__ == "__main__":
    compute_persistence_diagrams()