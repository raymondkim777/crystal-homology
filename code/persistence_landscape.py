import pickle
from gtda.diagrams import PersistenceLandscape
import numpy as np


def generate_landscape(diagrams: np.ndarray, n_layers=2, n_bins=100):
    transformer = PersistenceLandscape(n_layers=n_layers, n_bins=n_bins)
    transformed = transformer.fit_transform(diagrams)
    print("Homology Dimensions:", transformer.homology_dimensions_)
    print(transformed.shape)
    # fit_transform_plot
    fig = transformer.plot(transformed, sample=120)
    fig.show()

    return transformed.flatten()
    # return transformed[0].flatten()