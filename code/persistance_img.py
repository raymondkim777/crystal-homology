import pickle
from gtda.diagrams import PersistenceImage
import numpy as np

def generate_image(diagram, n_bins=50, sigma=1.0):
    transformer = PersistenceImage(n_bins=n_bins, sigma=sigma)
    image = transformer.fit_transform(diagram)
    return image[0].flatten()