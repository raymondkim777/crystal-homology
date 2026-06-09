import numpy as np
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from gtda.homology import FlagserPersistence
from gtda.plotting import plot_diagram
# from vectorizers_batch import diagrams_isolate_dim, remove_diagram_padding


CRYSTAL_SYSTEMS = [
    'triclinic', 
    'monoclinic', 
    'trigonal', 
    'hexagonal', 
    'orthorhombic', 
    'tetragonal', 
    'cubic'
]


# with open(f'data/diagrams/triclinic.pkl', 'rb') as file:
#     diagrams = pickle.load(file)

# keys_list = list(diagrams.keys())
# diagram_array = np.stack(tuple(diagrams.values()), axis=0) 
# diagrams_dims = diagrams_isolate_dim(diagram_array)

# diagram_list = diagrams_dims[2]
# has_empty = any(len(d) == 0 for d in diagram_list)
# print(f"Contains empty diagrams: {has_empty}")


with open(f'data/graphs/cubic.pkl', 'rb') as file:
    graphs = pickle.load(file)

print(graphs['mp-97'].nodes(data=True))
for node, data in graphs['mp-97'].nodes(data=True):
    print(type(node))

for node in graphs['mp-97'].nodes:
    print(graphs['mp-97'].nodes[node]['specie'].number)
print()
print(graphs['mp-97'].edges(data=True))