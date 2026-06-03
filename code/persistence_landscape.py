import pickle
from gtda.diagrams import PersistenceLandscape
import numpy as np


def generate_landscape(diagrams, n_layers=2, n_bins=100):
    transformer = PersistenceLandscape(n_layers=n_layers, n_bins=n_bins)
    transformed = transformer.fit_transform([diagrams])
    return transformed[0].flatten()