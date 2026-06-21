import os
import json
import networkx as nx
import matplotlib.pyplot as plt


CRYSTAL_SYSTEMS = [
    'cubic', 
    'hexagonal', 
    'monoclinic', 
    'orthorhombic', 
    'tetragonal', 
    'triclinic', 
    'trigonal'
]

FIELDS = [
    "material_id", 
    "symmetry", 
    "structure", 
    # Chemist recommended fields
    # 'phonon_IDs',     # idk about this one
    'bulk_modulus', 
    # 'dos',            # use mpr.get_dos_by_material_id("mp-id") for DOS curve
    'bandstructure', 
    'band_gap', 
    # 'cbm', 
    # 'vbm', 
    'efermi', 
    'is_gap_direct'
]

PREDICT = [
    'system',
    'bulk_modulus', 
    'dos',              # future task
    'bandstructure', 
    'band_gap', 
    'efermi', 
    'is_gap_direct'
]

DIMENSION_CNT = 3
MAX_DIST = 12.43843407284584  # computed from find_max_dist()
MAX_NBR = 38


def get_num_cpus(default=1):
    if "SLURM_CPUS_PER_TASK" in os.environ:
        return int(os.environ["SLURM_CPUS_PER_TASK"])
    return default


def open_write_file(dir_path, file_name):
    """Opens a file for writing, or creates new file if file doesn't exist."""
    file_path = os.path.join(dir_path, file_name)
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    return file_path


def plot_nxgraph(graph: nx.DiGraph) -> None:
    # Draw the graph with labels
    pos = nx.spring_layout(graph)

    nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=800)
    edge_labels = nx.get_edge_attributes(graph, "weight")
    formatted_labels = {edge: f"{weight:.3f}" for edge, weight in edge_labels.items()}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=formatted_labels)
    
    # Display the plot
    plt.show()


def get_max_dist():
    with open('data/bounds.json', 'r') as f:
        return json.load(f)['max_bond_dist']