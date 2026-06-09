import numpy as np
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from gtda.homology import FlagserPersistence
from gtda.plotting import plot_diagram
# from vectorizers_batch import generate_landscape_gtda, generate_image_gtda, generate_landscape_gudhi, generate_image_gudhi
# from vectorizers import generate_landscape_gudhi, generate_image_gudhi
from pymatgen.io.cif import CifParser
from pymatgen.core.structure import Structure


def plot_persistence_diagram(diagram) -> None:
    fig = plot_diagram(diagram)
    fig.show()


def get_structures_from_cif(filepath: str) -> list:
    cif_parser = CifParser(filepath)
    structures = cif_parser.parse_structures()
    for struct in structures:
        check_result = cif_parser.check(struct)
        if check_result is not None:
            print(f"CIF Error: {filepath}")
            print(f"Error Message: {check_result}")
            raise ValueError(f"Struct contained in {filepath} is invalid")
    # TODO: look into which crystals have multiple structures
    return structures
    # return Structure.from_file(filepath)


def test():
    structure_filename = f"data/cif/cubic/mp-97.cif"
    structures = get_structures_from_cif(structure_filename)
    struct = structures[0]
    all_nbrs = struct.get_all_neighbors(8, include_index=True)
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
    nbr = all_nbrs[0]
    print(nbr[0])
    # nbr[i] == (site, distance, index, image)
    print(type(nbr[0][1]))
    print(nbr[0][2])

    nbr_fea = []
    for nbr in all_nbrs:
        nbr_fea.append(list(map(lambda x: x[1], nbr)))
    # print(nbr_fea)


    # print("asdfasdf\n")
    # crystal = Structure.from_file(structure_filename)
    # print(crystal)
    # print(crystal[0].specie.number)


def test2():
    a = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
    b = np.array([4, 2, 5])
    print(a[b])



if __name__ == "__main__":
    test()