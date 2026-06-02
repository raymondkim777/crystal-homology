import os
from tqdm import tqdm
import pickle
from utils import open_write_file
from pymatgen.io.cif import CifParser
from pymatgen.core.structure import Structure
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.analysis.graphs import StructureGraph

import matplotlib.pyplot as plt
import networkx as nx


CRYSTAL_SYSTEMS = [
    'triclinic', 
    'monoclinic', 
    'trigonal', 
    'hexagonal', 
    'orthorhombic', 
    'tetragonal', 
    'cubic'
]
CIF_DIRECTORY = "data/cif"
GRAPH_DIRECTORY = "data/graphs"


def get_structures_from_cif(filepath: str):
    cif_parser = CifParser(filepath)
    structures = cif_parser.parse_structures()
    for struct in structures:
        check_result = cif_parser.check(struct)
        if check_result is not None:
            print(f"CIF Error: {filepath}")
            print(f"Error Message: {check_result}")
    return structures
    # return Structure.from_file(filepath)


def construct_crystalnn_graph():
    # first pass --> convert multigraph into simple graph
    # use minimum euclidean distance for edge weights
    # remove self connections
    
    crystalnn = CrystalNN(distance_cutoffs=None, x_diff_weight=0, porous_adjustment=False)
    for system in CRYSTAL_SYSTEMS:

        system_graphs = dict()
        
        print(f"Fetching CIF files of {system} system...")
        cif_files = []

        # os.scandir() returns an iterator of DirEntry objects
        with os.scandir(f"{CIF_DIRECTORY}/{system}") as entries:
            for entry in entries:
                if not entry.is_file():
                    continue
                cif_files.append(entry.name)
                    
        print(f"Creating graphs of {system} system...")

        for filename in tqdm(cif_files):
            structure_file = f"{CIF_DIRECTORY}/{system}/{filename}"
            structures = get_structures_from_cif(structure_file)

            # transform into graph
            bonded_graph = crystalnn.get_bonded_structure(structures[0])
            nx_graph = nx.DiGraph(bonded_graph.graph)

            # remove self connections
            for node in nx_graph.nodes:
                if nx_graph.has_edge(*(node, node)):
                    nx_graph.remove_edge(*(node, node))

            # add edge weights (min. Euc. distances)
            distance_matrix = structures[0].distance_matrix
            distances_dict = dict()
            for edge in nx_graph.edges:
                distances_dict[edge] = distance_matrix[edge[0], edge[1]]
            nx.set_edge_attributes(nx_graph, values=distances_dict, name='weight')

            system_graphs[filename[:-4]] = nx_graph
        
        # save system_graphs as pickle
        graph_filepath = open_write_file(GRAPH_DIRECTORY, f'{system}.pkl')
        with open(graph_filepath, 'wb') as f:
            pickle.dump(system_graphs, f)


def test_crystalnn():
    # parameters selected for structural bonds, not chemical bonds
    crystalnn = CrystalNN(distance_cutoffs=None, x_diff_weight=0, porous_adjustment=False)

    structure_file = "data/cif/cubic/mp-97.cif"
    structures = get_structures_from_cif(structure_file)
    print(len(structures))

    # site_index = 0
    # neighbors = crystalnn.get_nn_info(structures[0], site_index)
    # print(neighbors)

    # for neighbor in neighbors:
    #     print(f"Species: {neighbor['site'].specie}")
    #     print(f"Distance: {neighbor['weight']} Å") # weight gives the fractional site occupancy or bond presence 

    bonded_graph = crystalnn.get_bonded_structure(structures[0])
    # print(bonded_graph)
    # print(type(bonded_graph))

    nx_graph = bonded_graph.graph
    nx_graph = nx.DiGraph(nx_graph)

    # remove self connections
    for node in nx_graph.nodes:
        nx_graph.remove_edge(*(node, node))

    # add edge weights (min. Euc. distances)
    distance_matrix = structures[0].distance_matrix
    distances_dict = dict()
    for edge in nx_graph.edges:
        print(type(edge))
        distances_dict[edge] = distance_matrix[edge[0], edge[1]]
    nx.set_edge_attributes(nx_graph, values=distances_dict, name='weight')
    # distance = structures[0].get_distance(0, 1)
    # print(f"distance b/w 0, 1:", distance)
    # dist_matrix = structure.distance_matrix
    # dist_matrix[i, j]:.3f

    print(nx_graph)
    print(type(nx_graph))
    print(nx_graph.edges)

    print("neighbors:", nx_graph.neighbors(0))
    print("neighbors list:", list(nx_graph.neighbors(0)))
    print("edges:", nx_graph.edges([0]))

    # Draw the graph with labels
    pos = nx.spring_layout(nx_graph)

    nx.draw(nx_graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=800)
    edge_labels = nx.get_edge_attributes(nx_graph, "weight")
    formatted_labels = {edge: f"{weight:.3f}" for edge, weight in edge_labels.items()}
    nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=formatted_labels)
    
    # Display the plot
    # plt.show()

    graph_filepath = open_write_file('data/graphs', 'testgraph.pkl')
    with open(graph_filepath, 'wb') as f:
        pickle.dump(nx_graph, f)
    # nx.write_graphml(nx_graph, graph_filepath)


if __name__ == "__main__":
    # test_crystalnn()
    construct_crystalnn_graph()