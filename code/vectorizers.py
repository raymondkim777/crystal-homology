import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import gudhi.representations as gdr
import gtda.diagrams as gtd
from gudhi.representations import Landscape, PersistenceImage
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

IMAGE_DIRECTORY = "data/images"
DIMENSION_CNT = 3
MAX_DIST = 12.43843407284584  # computed from find_max_dist()


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--landscape', action='store_true', help='generates persistence landscapes')
    parser.add_argument('--image', action='store_true', help='generates persistence images')
    return parser.parse_args()


def collect_system_diagrams(system: str) -> dict:
    '''Collects all persistence diagrams from .pkl files from one system into one list.'''
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
    Same as 'process_one_diagram', but for a list of diagrams.
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


def persistence_image(
        resolution=[20, 20], 
        bandwidth=1.0, 
    ):
    '''
    Fits image transformers to all diagrams per dimension. 
    Then generates persistence images for every diagram for each dimension. 
    '''
    # persistence diagrams for all systems (giotto-tda format)
    all_diagrams = collect_all_diagrams()
    
    # convert all diagrams into gudhi format, separate into dimensions
    processed_diagrams = process_all_diagrams(all_diagrams)

    # define and fit image classes
    transformers = []  # one per dimension
    for dim in range(DIMENSION_CNT):
        transformer = PersistenceImage(bandwidth=bandwidth, resolution=resolution)
        transformer.fit(processed_diagrams[dim])
        transformers.append(transformer)
    
    for system in CRYSTAL_SYSTEMS:
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




if __name__ == "__main__":
    args = _parse_args()

    if args.landscape:
        pass
    if args.image:
        persistence_image()