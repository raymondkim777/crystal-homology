import numpy as np
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from gtda.homology import FlagserPersistence
from gtda.plotting import plot_diagram
from persistence_landscape import generate_landscape


def plot_persistence_diagram(diagram) -> None:
    fig = plot_diagram(diagram)
    fig.show()


with open('data/diagrams/cubic.pkl', 'rb') as file:
    diagrams = pickle.load(file)

index = 120
keys_list = list(diagrams.keys())

# print(type(diagrams[keys_list[index]]))
# print(diagrams[keys_list[index]])
# print(diagrams[keys_list[index]].shape)
# print(keys_list[index])
plot_persistence_diagram(diagrams[keys_list[index]])

# print(tuple(diagrams.values()))
diagram_array = np.stack(tuple(diagrams.values()), axis=0) 

print(type(diagram_array))
print(diagram_array.shape)

landscape = generate_landscape(diagram_array)
print(landscape)