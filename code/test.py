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

