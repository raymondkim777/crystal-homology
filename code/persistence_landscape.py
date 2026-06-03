from gtda.diagrams import PersistenceLandscape
import numpy as np


def generate_landscape(
        diagrams: np.ndarray, 
        n_layers=2, 
        n_bins=100, 
        plot=False, 
        plot_index=None
    ):
    transformer = PersistenceLandscape(n_layers=n_layers, n_bins=n_bins)
    transformed = transformer.fit_transform(diagrams)
    
    print("Homology Dimensions:", transformer.homology_dimensions_)
    print("Transformed Shape:", transformed.shape)

    if not plot:
        return transformed.flatten()
    
    # plotting
    if plot_index is None:
        fig = transformer.plot(transformed)
    else:
        fig = transformer.plot(transformed, sample=plot_index)
    fig.show()

    return transformed.flatten()