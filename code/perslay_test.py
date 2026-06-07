import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchPersLay as tp
import gudhi.representations as gdr
from sklearn.preprocessing import MinMaxScaler


CRYSTAL_SYSTEMS = [
    'triclinic', 
    'monoclinic', 
    'trigonal', 
    'hexagonal', 
    'orthorhombic', 
    'tetragonal', 
    'cubic'
]
DIMENSION_CNT = 3


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


def homogenize_shape(diagrams: np.ndarray):
    '''Homogenizes np shape for given diagram array (one dimension)'''
    pass


def test():
    # ! PERSLAY USAGE EXAMPLE
    constant = 1.0  # learnable
    power = 0.0

    weight = tp.PowerPerslayWeight(constant=constant, power=power)

    image_size = (5, 5)
    image_bnds = ((-0.5, 1.5), (-0.5, 1.5))
    variance = 0.1

    phi = tp.GaussianPerslayPhi(
        image_size=image_size,
        image_bnds=image_bnds,
        variance=variance,  # learnable
    )

    perm_op = torch.sum
    rho = nn.Identity()

    perslay = tp.Perslay(weight=weight, phi=phi, perm_op=perm_op, rho=rho)

    # example diagrams

    all_diagrams = collect_all_diagrams()  # list
    processed_diagrams = process_all_diagrams(all_diagrams)
    print(processed_diagrams[0][0])
    # [[h0_diagram1, h0_diagram2, ...], [h1_diagram1, ...], ...] 
    #         where each Hn diagram is [[b, d], ...]

    # have to homogenize shape if feeding in multiple

    scaler = gdr.DiagramScaler(use=True, scalers=[([0, 1], MinMaxScaler())])
    diagrams = scaler.fit_transform([processed_diagrams[0][0]])
    diagrams = torch.from_numpy(np.array(diagrams, dtype=np.float32))

    print(perslay(diagrams))


if __name__ == "__main__":
    test()