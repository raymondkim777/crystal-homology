import os
import numpy as np
import networkx as nx
from tqdm import tqdm
import pickle
from utils import open_write_file, plot_graph
from gtda.homology import FlagserPersistence
from gtda.plotting import plot_diagram


CRYSTAL_SYSTEMS = [
    'triclinic', 
    'monoclinic', 
    'trigonal', 
    'hexagonal', 
    'orthorhombic', 
    'tetragonal', 
    'cubic'
]
GRAPH_DIRECTORY = "data/graphs"
DIAGRAM_DIRECTORY = "data/diagrams"
MAX_DIST = 12.43843407284584


def unpack_all_graphs() -> dict:
    graph_dict = dict()
    for system in CRYSTAL_SYSTEMS:
        with open(f'data/graphs/{system}.pkl', 'rb') as file:
            graph_dict[system] = pickle.load(file)
    return graph_dict


def find_max_dist(graph_dict: dict) -> np.float64:    # max_finite_dist = my_matrix[my_matrix != np.inf].max()
    max_dist = 0
    for system in CRYSTAL_SYSTEMS: 
        for graph in graph_dict[system].values():
            weights = nx.get_edge_attributes(graph, "weight").values()
            if len(weights) == 0:
                continue
            max_dist = max(max_dist, max(weights))
    return max_dist


def convert_graph_to_adj_mat(graph: nx.DiGraph) -> np.ndarray:
    # ? to_numpy_array can also handle multigraph weights --> extension?
    adj_mat = nx.to_numpy_array(graph, weight='weight', nonedge=np.inf)
    
    # set diagonals to 0 (just in case)
    for i in range(len(graph.nodes)):
        adj_mat[i][i] = 0
    return adj_mat


def plot_persistence_diagram(diagram) -> None:
    fig = plot_diagram(diagram)
    fig.show()


def analyze_homology(dims: tuple=(0, 1, 2)) -> None:
    
    print(f"Unpacking all graphs...")
    graph_dict = unpack_all_graphs()

    flagser = FlagserPersistence(
        homology_dimensions=dims,
        directed=True,
        filtration='max', 
        coeff=2, 
        max_edge_weight=MAX_DIST + 1,
        infinity_values=np.inf
    )

    for system in CRYSTAL_SYSTEMS:
        print(f"Computing PD for {system}...")
        graph_list = graph_dict[system].values()
        adj_mat_list = list(map(convert_graph_to_adj_mat, graph_list))

        diagrams = flagser.fit_transform(adj_mat_list)

        # print diagram info
        print(f"diagram cnt: {len(diagrams)}")
        graph_diagram = diagrams[0]
        print("Persistence Diagram Shape:", graph_diagram.shape)
        print("Points (Birth, Death, Homology Dimension):\n", graph_diagram)
    
        # save diagrams dict as pickle
        diag_filepath = open_write_file(DIAGRAM_DIRECTORY, f'{system}.pkl')
        with open(diag_filepath, 'wb') as f:
            pickle.dump(diagrams, f)
    

def test():
    with open('data/graphs/cubic.pkl', 'rb') as file:
        graph_list = pickle.load(file)
    
    material_id = 'mp-97'
    graph = graph_list[material_id]

    print(find_max_dist(unpack_all_graphs()))


if __name__ == "__main__":
    # test()
    analyze_homology()