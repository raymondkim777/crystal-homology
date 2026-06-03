from gtda.diagrams import PersistenceLandscape
import numpy as np

class LandscapeGenerator:
    def __init__(self, n_layers=5, resolution=100):
        self.transformer = PersistenceLandscape(n_layers= n_layers, resolution=resolution)
    
    def generate(self, diagrams):

        transformed = self.transformer.fit_transform([diagrams])
        return transformed[0].flatten()