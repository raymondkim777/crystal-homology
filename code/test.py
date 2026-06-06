import numpy as np
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from gtda.homology import FlagserPersistence
from gtda.plotting import plot_diagram
# from vectorizers_batch import generate_landscape_gtda, generate_image_gtda, generate_landscape_gudhi, generate_image_gudhi
# from vectorizers import generate_landscape_gudhi, generate_image_gudhi


def plot_persistence_diagram(diagram) -> None:
    fig = plot_diagram(diagram)
    fig.show()


with open('data/diagrams/monoclinic.pkl', 'rb') as file:
    diagrams = pickle.load(file)

index = 800
keys_list = list(diagrams.keys())

# print(type(diagrams[keys_list[index]]))
# print(diagrams[keys_list[index]])
# print(diagrams[keys_list[index]].shape)
# print(keys_list[index])
plot_persistence_diagram(diagrams[keys_list[index]])

print(tuple(diagrams.values()))
diagram_array = np.stack(tuple(diagrams.values()), axis=0) 
