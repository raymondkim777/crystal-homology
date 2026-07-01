import argparse
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Directs Matplotlib to write to a file, not a GUI window
import gudhi.representations as gdr
import gtda.diagrams as gtd
from gudhi.representations import Landscape, PersistenceImage

from concurrent.futures import ProcessPoolExecutor
from compute_diagrams import plot_persistence_diagram
from utils import CRYSTAL_SYSTEMS, DIMENSION_CNT, get_num_cpus, open_write_file, get_max_dist


# DATA_DIRECTORY = "data/pretrain"
# DATA_DIRECTORY = "data/abs"
# DIAGRAM_DIRECTORY = f"{DATA_DIRECTORY}/diagrams"
# LANDSCAPE_DIRECTORY = f"{DATA_DIRECTORY}/landscapes"
# IMAGE_DIRECTORY = f"{DATA_DIRECTORY}/images"
# MAX_DIST = get_max_dist(DATA_DIRECTORY) 

DATA_DIRECTORY = None
DIAGRAM_DIRECTORY = None
LANDSCAPE_DIRECTORY = None
IMAGE_DIRECTORY = None
MAX_DIST = None

LANDSCAPE_TRANSFORMERS = None
IMAGE_TRANSFORMERS = None

# landscape
LA_LAYER = 5
LA_RESOLUTION = 100

# images
IM_BANDWIDTH = 1.0
IM_RESOLUTION = [20, 20]


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--abs', action='store_true', help='vectorizes absorption PDs')
    parser.add_argument('--plqy', action='store_true', help='Only focuses separately on data/plqys')
    parser.add_argument('--landscape', action='store_true', help='generates persistence landscapes')
    parser.add_argument('--image', action='store_true', help='generates persistence images')
    parser.add_argument('--example', action='store_true', help='display some examples')
    return parser.parse_args()


def collect_system_diagrams(system: str) -> dict:
    '''Collects persistence diagrams from .pkl files from one system as a dict.'''
    with open(f'{DIAGRAM_DIRECTORY}/{system}.pkl', 'rb') as file:
        diagrams = pickle.load(file)
    return diagrams


def collect_all_diagrams() -> list:
    '''Collects all persistence diagrams from .pkl files from all systems into one list.'''
    all_diagrams = []
    for system in CRYSTAL_SYSTEMS:
        with open(f'{DIAGRAM_DIRECTORY}/{system}.pkl', 'rb') as file:
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
    # ! sample_range is already provided, so no need to fit
    # # persistence diagrams for all systems (giotto-tda format)
    # all_diagrams = collect_all_diagrams()
    
    # # convert all diagrams into gudhi format, separate into dimensions
    # processed_diagrams = process_all_diagrams(all_diagrams)

    # define and fit landscape classes
    print("fitting landscape transformers...")

    dummy_diagrams = [np.empty((0, 2))]

    transformers = []  # one per dimension
    for dim in range(DIMENSION_CNT):
        transformer = Landscape(
            num_landscapes=num_landscapes, 
            resolution=resolution, 
            sample_range=[0, MAX_DIST]
        )
        # transformer.fit(processed_diagrams[dim])
        transformer.fit(dummy_diagrams)  # to create grid_
        transformers.append(transformer)
    return transformers


def fit_image_transformers(bandwidth, resolution):
    '''
    For each dimension, fits image transformers to all system persistence diagrams. 
    '''
    # persistence diagrams for all systems (giotto-tda format)
    all_diagrams = collect_all_diagrams()
    
    # convert all diagrams into gudhi format, separate into dimensions
    processed_diagrams = process_all_diagrams(all_diagrams)

    # define and fit image classes
    print("fitting image transformers...")

    transformers = []  # one per dimension
    for dim in range(DIMENSION_CNT):
        transformer = PersistenceImage(bandwidth=bandwidth, resolution=resolution)
        transformer.fit(processed_diagrams[dim])
        transformers.append(transformer)
    return transformers


def init_landscape_transformers(transformers):
    global LANDSCAPE_TRANSFORMERS
    LANDSCAPE_TRANSFORMERS = transformers


def init_image_transformers(transformers):
    global IMAGE_TRANSFORMERS
    IMAGE_TRANSFORMERS = transformers


def compute_landscapes_for_system(args):
    '''
    Generates persistence landscapes for one system for each dimension.
    '''
    system = args

    # print(f"computing landscapes for {system} system...")
    # process all diagrams in system
    system_diagrams = collect_system_diagrams(system)
    keys_list = list(system_diagrams.keys())
    if len(keys_list) == 0:
        return system, dict()
    diagrams_by_dims = process_all_diagrams(list(system_diagrams.values()))

    # generate persistence landscapes for all diagrams for each dimension
    landscapes_by_dim = []  # [[h0_land1, h0_land2, ...], [h1_land1, ...], ...]
    for dim in range(DIMENSION_CNT):
        landscapes_by_dim.append(LANDSCAPE_TRANSFORMERS[dim].transform(diagrams_by_dims[dim]))
    
    # match to id and organize by id -> dim
    system_landscapes = dict()
    for i in range(len(keys_list)):
        key = keys_list[i]
        system_landscapes[key] = {
            dim: landscapes_by_dim[dim][i] 
            for dim in range(DIMENSION_CNT)
        }
    
    return system, system_landscapes


def persistence_landscape(
        num_landscapes=LA_LAYER, 
        resolution=LA_RESOLUTION,
    ):
    '''
    Generates persistence landscapes for every system for each dimension.
    Uses multiprocessing. 
    '''
    n_workers = get_num_cpus()
    landscape_transformers = fit_landscape_transformers(num_landscapes, resolution)
    
    print(f"Computing landscapes with {n_workers} workers...")
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=init_landscape_transformers,
        initargs=(landscape_transformers,),
    ) as executor:
        results = executor.map(compute_landscapes_for_system, CRYSTAL_SYSTEMS)

        for system, system_landscapes in tqdm(results, total=len(CRYSTAL_SYSTEMS)):
            # save system images
            landscape_path = open_write_file(LANDSCAPE_DIRECTORY, f"{system}.pkl")
            with open(landscape_path, 'wb') as f:
                pickle.dump(system_landscapes, f)


def pad_bounds(bound_x, bound_y, eps=0.001):
    if bound_x == bound_y:
        bound_x -= eps
        bound_y -= eps
    return (bound_x, bound_y)


def save_image_transformer_bounds(image_transformers):
    image_bnds_list = []
    for dim in range(DIMENSION_CNT):
        bounds = image_transformers[dim].im_range_fixed_
        bounds_tuple = (pad_bounds(bounds[0], bounds[1]), pad_bounds(bounds[2], bounds[3]))
        image_bnds_list.append(bounds_tuple)
    
    # save JSON
    file_path = open_write_file(f'{DATA_DIRECTORY}', 'image_bounds.pkl')
    with open(file_path, "wb") as f:
        pickle.dump(image_bnds_list, f)


def compute_images_for_system(args):
    '''
    Generates persistence images for one system for each dimension. 
    '''
    system = args

    # print(f"computing images for {system} system...")
    # process all diagrams in system
    system_diagrams = collect_system_diagrams(system)
    keys_list = list(system_diagrams.keys())
    if len(keys_list) == 0:
        return system, dict()
    diagrams_by_dims = process_all_diagrams(list(system_diagrams.values()))

    # generate persistence images for all diagrams for each dimension
    images_by_dim = []  # [[h0_image1, h0_image2, ...], [h1_image1, ...], ...]
    for dim in range(DIMENSION_CNT):
        images_by_dim.append(IMAGE_TRANSFORMERS[dim].transform(diagrams_by_dims[dim]))
    
    # match to id and organize by id -> dim
    system_images = dict()
    for i in range(len(keys_list)):
        key = keys_list[i]
        system_images[key] = {
            dim: images_by_dim[dim][i] 
            for dim in range(DIMENSION_CNT)
        }
    
    # # save system images
    # image_path = open_write_file(IMAGE_DIRECTORY, f"{system}.pkl")
    # with open(image_path, 'wb') as f:
    #     pickle.dump(system_images, f)
    return system, system_images


def persistence_image(
        bandwidth=IM_BANDWIDTH, 
        resolution=IM_RESOLUTION, 
    ):
    '''
    Generates persistence images for every system for each dimension. 
    Uses multiprocessing. 
    '''
    n_workers = get_num_cpus()
    image_transformers = fit_image_transformers(bandwidth, resolution)

    # save image bounds
    save_image_transformer_bounds(image_transformers)

    print(f"Computing images with {n_workers} workers...")
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=init_image_transformers,
        initargs=(image_transformers,),
    ) as executor:
        results = executor.map(compute_images_for_system, CRYSTAL_SYSTEMS)

        for system, system_images in tqdm(results, total=len(CRYSTAL_SYSTEMS)):
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
    with open(f'{LANDSCAPE_DIRECTORY}/{system}.pkl', 'rb') as file:
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
    # plt.show()
    plt.savefig(f'{LANDSCAPE_DIRECTORY}/example_gudhi.png')


def plot_image(system: str, mat_id: str, dim: int=0):
    '''
    Plots persistence diagram and image of given material id in given system, for given dimension. 
    '''
    diagram = collect_system_diagrams(system)[mat_id]
    plot_persistence_diagram(diagram)

    transformers = fit_image_transformers(bandwidth=IM_BANDWIDTH, resolution=IM_RESOLUTION)
    with open(f'{IMAGE_DIRECTORY}/{system}.pkl', 'rb') as file:
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
    # plt.show()
    plt.savefig(f'{IMAGE_DIRECTORY}/example_gudhi.png')


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
    assert not args.abs or not args.plqy, "Can only choose one of abs/plqy"

    DATA_DIRECTORY = "data/pretrain"
    if args.abs:
        DATA_DIRECTORY = "data/abs"
    if args.plqy:
        DATA_DIRECTORY = "data/plqy"

    DIAGRAM_DIRECTORY = f"{DATA_DIRECTORY}/diagrams_g"
    LANDSCAPE_DIRECTORY = f"{DATA_DIRECTORY}/landscapes_g"
    IMAGE_DIRECTORY = f"{DATA_DIRECTORY}/images_g"
    MAX_DIST = get_max_dist(DATA_DIRECTORY) 

    if args.landscape:
        persistence_landscape()
        if args.example:
            plot_landscape_gtda('triclinic', 'mp-2981')
            plot_landscape('triclinic', 'mp-2981', dim=0)
    if args.image:
        persistence_image()
        if args.example:
            plot_image('triclinic', 'mp-2981', dim=1)