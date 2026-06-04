import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import gudhi.representations as gdr
import gtda.diagrams as gtd
from gudhi.representations import Landscape, PersistenceImage

from compute_diagrams import plot_persistence_diagram
from utils import open_write_file

CRYSTAL_SYSTEMS = [
    'triclinic', 
    'monoclinic', 
    'trigonal', 
    'hexagonal', 
    'orthorhombic', 
    'tetragonal', 
    'cubic'
]

LANDSCAPE_DIRECTORY = "data/landscapes"
IMAGE_DIRECTORY = "data/images"
DIMENSION_CNT = 3
MAX_DIST = 12.43843407284584  # computed from find_max_dist()

# landscape
LA_LAYER = 5
LA_RESOLUTION = 100

# images
IM_BANDWIDTH = 1.0
IM_RESOLUTION = [20, 20]


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--landscape', action='store_true', help='generates persistence landscapes')
    parser.add_argument('--image', action='store_true', help='generates persistence images')
    parser.add_argument('--example', action='store_true', help='display some examples')
    return parser.parse_args()


def collect_system_diagrams(system: str) -> dict:
    '''Collects persistence diagrams from .pkl files from one system as a dict.'''
    with open(f'data/diagrams/{system}.pkl', 'rb') as file:
        diagrams = pickle.load(file)
    return diagrams


def collect_all_diagrams() -> list:
    '''Collects all persistence diagrams from .pkl files from all systems into one list.'''
    all_diagrams = []
    for system in CRYSTAL_SYSTEMS:
        with open(f'data/diagrams/{system}.pkl', 'rb') as file:
            diagrams = pickle.load(file)
        all_diagrams += list(diagrams.values())
    return all_diagrams


def remove_diagram_padding(diagram, eps=1e-12):
    '''Removes all infinite values & (b, d) such that b == d'''
    diagram = np.asarray(diagram, dtype=float)

    finite_mask = np.isfinite(diagram).all(axis=1)
    persistence_mask = (diagram[:, 1] - diagram[:, 0]) > eps
    final_diagram = diagram[finite_mask & persistence_mask]
    # NOTE: final diagram may be empty
    return final_diagram


def process_one_diagram(diagram: np.ndarray):
    '''
    Removes all diagram giotto-tda padding (b == d), organizes triplets into
    separate dimensions, and removes dimension field.
    Input: diagram = [[b, d, dim], ...]
    Output: [h0_diagram, h1_diagram, h2_diagram] 
            where each diagram is [[b, d], ...] for that dimension
    '''
    diagram_dim = []
    for dim in range(DIMENSION_CNT):
        triplets_in_dim = diagram[diagram[:, 2] == dim]
        doubles_in_dim = triplets_in_dim[:, :2]
        final_diagram = remove_diagram_padding(doubles_in_dim)
        diagram_dim.append(final_diagram)
    return diagram_dim


def process_all_diagrams(diagrams: list):
    '''
    Removes all diagram giotto-tda padding (b == d), organizes triplets into
    separate dimensions, and removes dimension field. Applied to list of diagrams.
    Input: [diagram1, diagram2, ...] where diagram = [[b, d, dim], ...]
    Output: [[h0_diagram1, h0_diagram2, ...], [h1_diagram1, ...], ...] 
            where each Hn diagram is [[b, d], ...]
    '''
    dimension_array = []
    for dim in range(DIMENSION_CNT):
        dim_diagrams = []
        for diagram in diagrams:
            triplets_in_dim = diagram[diagram[:, 2] == dim]
            doubles_in_dim = triplets_in_dim[:, :2]
            final_diagram = remove_diagram_padding(doubles_in_dim)
            dim_diagrams.append(final_diagram)
        dimension_array.append(dim_diagrams)
    return dimension_array


def fit_landscape_transformers(num_landscapes, resolution):
    '''
    For each dimension, fits landscape transformers to all system persistence diagrams. 
    '''
    # persistence diagrams for all systems (giotto-tda format)
    print("processing persistence diagrams...")
    all_diagrams = collect_all_diagrams()
    
    # convert all diagrams into gudhi format, separate into dimensions
    processed_diagrams = process_all_diagrams(all_diagrams)

    # define and fit landscape classes
    print("fitting landscape transformer...")

    transformers = []  # one per dimension
    for dim in range(DIMENSION_CNT):
        transformer = Landscape(
            num_landscapes=num_landscapes, 
            resolution=resolution, 
            sample_range=[0, MAX_DIST]
        )
        transformer.fit(processed_diagrams[dim])
        transformers.append(transformer)
    return transformers


def fit_image_transformers(bandwidth, resolution):
    '''
    For each dimension, fits image transformers to all system persistence diagrams. 
    '''
    # persistence diagrams for all systems (giotto-tda format)
    print("processing persistence diagrams...")
    all_diagrams = collect_all_diagrams()
    
    # convert all diagrams into gudhi format, separate into dimensions
    processed_diagrams = process_all_diagrams(all_diagrams)

    # define and fit image classes
    print("fitting image transformer...")

    transformers = []  # one per dimension
    for dim in range(DIMENSION_CNT):
        transformer = PersistenceImage(bandwidth=bandwidth, resolution=resolution)
        transformer.fit(processed_diagrams[dim])
        transformers.append(transformer)
    return transformers



def persistence_landscape(
        num_landscapes=LA_LAYER, 
        resolution=LA_RESOLUTION,
    ):
    '''
    Generates persistence landscapes for every system persistence diagram for each dimension. 
    '''
    transformers = fit_landscape_transformers(num_landscapes, resolution)
    
    for system in CRYSTAL_SYSTEMS:
        print(f"computing landscapes for {system} system...")
        # process all diagrams in system
        system_diagrams = collect_system_diagrams(system)
        keys_list = list(system_diagrams.keys())
        diagrams_by_dims = process_all_diagrams(list(system_diagrams.values()))

        # generate persistence landscapes for all diagrams for each dimension
        landscapes_by_dim = []  # [[h0_land1, h0_land2, ...], [h1_land1, ...], ...]
        for dim in range(DIMENSION_CNT):
            landscapes_by_dim.append(transformers[dim].transform(diagrams_by_dims[dim]))
        
        # match to id and organize by id -> dim
        system_landscapes = dict()
        for i in range(len(keys_list)):
            key = keys_list[i]
            system_landscapes[key] = {
                dim: landscapes_by_dim[dim][i] 
                for dim in range(DIMENSION_CNT)
            }
        
        # save system images
        landscape_path = open_write_file(LANDSCAPE_DIRECTORY, f"{system}.pkl")
        with open(landscape_path, 'wb') as f:
            pickle.dump(system_landscapes, f)


def persistence_image(
        bandwidth=IM_BANDWIDTH, 
        resolution=IM_RESOLUTION, 
    ):
    '''
    Generates persistence images for every system persistence diagram for each dimension. 
    '''
    transformers = fit_image_transformers(bandwidth, resolution)
    
    for system in CRYSTAL_SYSTEMS:
        print(f"computing images for {system} system...")
        # process all diagrams in system
        system_diagrams = collect_system_diagrams(system)
        keys_list = list(system_diagrams.keys())
        diagrams_by_dims = process_all_diagrams(list(system_diagrams.values()))

        # generate persistence images for all diagrams for each dimension
        images_by_dim = []  # [[h0_image1, h0_image2, ...], [h1_image1, ...], ...]
        for dim in range(DIMENSION_CNT):
            images_by_dim.append(transformers[dim].transform(diagrams_by_dims[dim]))
        
        # match to id and organize by id -> dim
        system_images = dict()
        for i in range(len(keys_list)):
            key = keys_list[i]
            system_images[key] = {
                dim: images_by_dim[dim][i] 
                for dim in range(DIMENSION_CNT)
            }
        
        # save system images
        image_path = open_write_file(IMAGE_DIRECTORY, f"{system}.pkl")
        with open(image_path, 'wb') as f:
            pickle.dump(system_images, f)


def plot_landscape(system: str, mat_id: str, dim: int=0):
    '''
    Plots persistence diagram and landscape of given material id in given system, for given dimension. 
    '''
    diagram = collect_system_diagrams(system)[mat_id]
    plot_persistence_diagram(diagram)

    transformers = fit_landscape_transformers(num_landscapes=LA_LAYER, resolution=LA_RESOLUTION)
    with open(f'data/landscapes/{system}.pkl', 'rb') as file:
        landscapes = pickle.load(file)
    landscape_to_plot = landscapes[mat_id][dim]  # all three dimensions
    
    x_values = np.linspace(*transformers[dim].sample_range_fixed_, LA_RESOLUTION)
    plt.figure(figsize=(8, 5))  

    for i in range(LA_LAYER):
        y_values = landscape_to_plot[i * LA_RESOLUTION : (i + 1) * LA_RESOLUTION]
        plt.plot(x_values, y_values, label=f"Landscape {i+1}")

    plt.title(f"Persistence Landscape Dimension {dim}")
    plt.xlabel("Parameter $t$")
    plt.ylabel("$\lambda_k(t)$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_image(system: str, mat_id: str, dim: int=0):
    '''
    Plots persistence diagram and image of given material id in given system, for given dimension. 
    '''
    diagram = collect_system_diagrams(system)[mat_id]
    plot_persistence_diagram(diagram)

    transformers = fit_image_transformers(bandwidth=IM_BANDWIDTH, resolution=IM_RESOLUTION)
    with open(f'data/images/{system}.pkl', 'rb') as file:
        images = pickle.load(file)
    image_to_plot = images[mat_id][dim]  # all three dimensions
    
    img_matrix = image_to_plot.reshape(IM_RESOLUTION)
    plt.figure(figsize=(6, 6))
    plt.imshow(img_matrix, cmap='viridis', origin='lower', 
            extent=transformers[dim].im_range_fixed_, interpolation='nearest')
    plt.title(f"Persistence Image (Dimension {dim})")
    plt.xlabel("Birth")
    plt.ylabel("Death")
    plt.colorbar(label="Pixel Intensity")
    plt.show()


################   UNUSED FROM BELOW   ################


# ONLY USED FOR CROSS REFERENCE
def plot_landscape_gtda(
        system: str, 
        mat_id: str,
        n_layers=LA_LAYER, 
        n_bins=LA_RESOLUTION, 
    ):

    # all_diagrams = np.asarray(collect_all_diagrams())
    system_diagrams_dict = collect_system_diagrams(system)

    keys_list = list(system_diagrams_dict.keys())
    system_diagrams = list(system_diagrams_dict.values())

    transformer = gtd.PersistenceLandscape(n_layers=n_layers, n_bins=n_bins)
    transformed = transformer.fit_transform(system_diagrams)
    system_landscapes = {
        keys_list[i] : transformed[i]
        for i in range(len(keys_list))
    }
    fig = transformer.plot([system_landscapes[mat_id]])
    fig.show()


if __name__ == "__main__":
    args = _parse_args()

    if args.landscape:
        persistence_landscape()
        if args.example:
            plot_landscape_gtda('triclinic', 'mp-2856')
            plot_landscape('triclinic', 'mp-2856', dim=2)
    if args.image:
        persistence_image()
        if args.example:
            plot_image('triclinic', 'mp-2856', dim=1)