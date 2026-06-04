import numpy as np
import matplotlib.pyplot as plt
from gudhi.representations import Landscape
from gtda.diagrams import PersistenceLandscape
from gtda.diagrams import PersistenceImage


# DIMENSION_CNT = 3


# def diagrams_giotto_to_gudhi(diagrams: np.ndarray):
#     alt_diagrams = []
#     for diag in diagrams:
#         alt_diagrams.append(diag[:, [2, 0, 1]])
#     return np.array(alt_diagrams)


# def diagrams_isolate_dim(diagrams: np.ndarray):
#     # return (hom_dim_n) list of diagram array (1000)
#     # entry could be None if no such holes exist
#     dimension_array = []

#     for dim in range(DIMENSION_CNT):
#         dim_diagrams = []
#         for diagram in diagrams:
#             # rows (holes) in diagram with correct dim
#             dim_rows = diagram[diagram[:, 2] == dim]
#             dim_diagrams.append(dim_rows[:, :2])
#         dimension_array.append(dim_diagrams)
#     return dimension_array


# def generate_landscape_gudhi(
#         diagrams: np.ndarray,  # giotto-tda format (b, d, dim)
#         n_landscapes=5, 
#         resolution=100,
#     ):
#     diagrams_dims = diagrams_isolate_dim(diagrams)

#     # can only handle one homology dimension at a time
#     landscape = Landscape(
#         num_landscapes=n_landscapes, 
#         resolution=resolution)
    
#     all_landscapes = []
#     for dim in range(DIMENSION_CNT):
#         all_landscapes.append(landscape.fit_transform(diagrams_dims[dim]))
#     return np.array(all_landscapes)


def generate_landscape_gtda(
        diagrams: np.ndarray, 
        n_layers=5, 
        n_bins=100, 
        plot=False, 
        plot_index=None
    ):
    transformer = PersistenceLandscape(n_layers=n_layers, n_bins=n_bins)
    transformed = transformer.fit_transform(diagrams)
    
    print("Homology Dimensions:", transformer.homology_dimensions_)
    print("Transformed Shape:", transformed.shape)
    print(transformed)

    if not plot:
        return transformed.flatten()
    
    # plotting
    if plot_index is None:
        fig = transformer.plot(transformed)
    else:
        fig = transformer.plot(transformed, sample=plot_index)
    fig.show()

    return transformed.flatten()


def generate_image_gtda(
        diagrams: np.ndarray, 
        n_bins=100, 
        sigma=0.1, 
        plot=False, 
        plot_index=None,
        plot_dim=0,
    ):
    transformer = PersistenceImage(n_bins=n_bins, sigma=sigma)
    transformed = transformer.fit_transform(diagrams)

    print("Homology Dimensions:", transformer.homology_dimensions_)
    print("Transformed Shape:", transformed.shape)

    if not plot:
        return transformed.flatten()
    
    # plotting
    if plot_index is None:
        fig = transformer.plot(transformed)
    else:
        fig = transformer.plot(
            transformed, 
            sample=plot_index,
            homology_dimension_idx=plot_dim,
        )
    fig.show()

    return transformed.flatten()