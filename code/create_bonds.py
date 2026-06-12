import os
from tqdm import tqdm
import pickle
from utils import CRYSTAL_SYSTEMS, open_write_file, plot_nxgraph
from pymatgen.io.cif import CifParser
from pymatgen.analysis.local_env import CrystalNN
# from pymatgen.core.structure import Structure
# from pymatgen.analysis.graphs import StructureGraph
import matplotlib.pyplot as plt
import networkx as nx


CIF_DIRECTORY = "data/cif"
MULTIGRAPH_DIRECTORY = "data/graphs-multi"
GRAPH_DIRECTORY = "data/graphs"


def fetch_cif_filenames(system: str) -> list:
    print(f"Fetching CIF files of {system} system...")
    cif_files = []

    # os.scandir() returns an iterator of DirEntry objects
    with os.scandir(f"{CIF_DIRECTORY}/{system}") as entries:
        for entry in entries:
            if not entry.is_file():
                continue
            cif_files.append(entry.name)
    return cif_files


def get_structures_from_cif(filepath: str) -> list:
    cif_parser = CifParser(filepath)
    structures = cif_parser.parse_structures()
    for struct in structures:
        check_result = cif_parser.check(struct)
        if check_result is not None:
            print(f"CIF Error: {filepath}")
            print(f"Error Message: {check_result}")
            raise ValueError(f"Struct contained in {filepath} is invalid")
    # TODO: look into which crystals have multiple structures
    return structures
    # return Structure.from_file(filepath)


def construct_crystalnn_graph() -> None:
    # first pass --> convert multigraph into simple graph
    # use minimum euclidean distance for edge weights
    # remove self connections
    
    # parameters selected for structural bonds, not chemical bonds
    crystalnn = CrystalNN(
        distance_cutoffs=None, 
        x_diff_weight=0, 
        porous_adjustment=False,
        search_cutoff=12  # default 7, but ERROR: No Voronoi neighbors found for site
    )
    for system in CRYSTAL_SYSTEMS:

        system_multigraphs = dict()
        system_graphs = dict()
        cif_files = fetch_cif_filenames(system)
                    
        print(f"Creating graphs of {system} system...")
        for filename in tqdm(cif_files):
            structure_filename = f"{CIF_DIRECTORY}/{system}/{filename}"
            structures = get_structures_from_cif(structure_filename)
            
            # transform into multi-digraph
            nx_multigraph = crystalnn.get_bonded_structure(structures[0])

            # set bond distances as weight for each edge
            for u, v, key, data in nx_multigraph.graph.edges(keys=True, data=True):
                to_jimage = data['to_jimage']
                dist = structures[0].get_distance(u, v, jimage=to_jimage)
                nx_multigraph.graph.edges[u, v, key]['weight'] = dist

            # change species identifier to species object, not label
            for node in nx_multigraph.graph.nodes:
                nx_multigraph.graph.nodes[node]['specie'] = structures[0][node].specie

            # save multigraph (for CGCNN graph input)
            system_multigraphs[filename[:-4]] = nx_multigraph.graph

            # collapse multigraph into graph
            nx_graph = nx.DiGraph(nx_multigraph.graph)

            # remove self connections
            for node in nx_graph.nodes:
                if nx_graph.has_edge(*(node, node)):
                    nx_graph.remove_edge(*(node, node))

            # save edge multiplicites --> DEPR, CGCNN can handle multiple edges
            edge_mult_dict = dict()
            for edge in nx_multigraph.graph.edges:  # tuple
                edge_collapsed = (edge[0], edge[1])  # three entries, third one is unique key
                edge_mult_dict[edge_collapsed] = len(nx_multigraph.graph[edge[0]][edge[1]])

            # store edge multiplicities as edge attribute
            nx.set_edge_attributes(nx_graph, values=edge_mult_dict, name='multiplicity')

            # re-implement edge weights as min. distances of all relevant bonds
            distance_matrix = structures[0].distance_matrix
            distances_dict = dict()
            for edge in nx_graph.edges:
                distances_dict[edge] = distance_matrix[edge[0], edge[1]]
            nx.set_edge_attributes(nx_graph, values=distances_dict, name='weight')
            
            # remove to_jimage attribute
            for u, v, data in nx_graph.edges(data=True):
                data.pop("to_jimage", None)

            # save graph
            system_graphs[filename[:-4]] = nx_graph
        
        # save graphs as pickles

        multigraph_filepath = open_write_file(MULTIGRAPH_DIRECTORY, f'{system}.pkl')
        with open(multigraph_filepath, 'wb') as f:
            pickle.dump(system_multigraphs, f)
        
        graph_filepath = open_write_file(GRAPH_DIRECTORY, f'{system}.pkl')
        with open(graph_filepath, 'wb') as f:
            pickle.dump(system_graphs, f)


def check_structures() -> None:
    for system in CRYSTAL_SYSTEMS:
        struct_mult_list = []
        cif_files = fetch_cif_filenames(system)
        for filename in tqdm(cif_files):
            structure_filename = f"{CIF_DIRECTORY}/{system}/{filename}"
            structures = get_structures_from_cif(structure_filename)
            if len(structures) > 1:
                struct_mult_list.append(filename[:-4])
        filepath = open_write_file(f'data/struct-issues', f"{system}")
        with open(filepath, 'w') as f:
            f.write("\n".join(struct_mult_list))


def test_crystalnn() -> None:
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
    print(bonded_graph.graph.edges(data=True))
    print(type(bonded_graph.graph.edges[0, 4, 0]['to_jimage']))
    return

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

    # plot_nxgraph(nx_graph)

    graph_filepath = open_write_file('data/graphs', 'testgraph.pkl')
    with open(graph_filepath, 'wb') as f:
        pickle.dump(nx_graph, f)
    # nx.write_graphml(nx_graph, graph_filepath)


if __name__ == "__main__":
    # test_crystalnn()
    construct_crystalnn_graph()
    # check_structures()