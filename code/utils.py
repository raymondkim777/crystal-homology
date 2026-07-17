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
    'bandstructure', 
    'band_gap', 
    'efermi', 
    'is_gap_direct',
]

PREDICT = [
    'system',
    'direct_gap',
    'band_gap', 
    'efermi', 
    'is_gap_direct',
]

ABS_PREDICT = [
    'integrated_absorption',
    'integrated_absorption_visible',
    'average_absorption_visible',
]

PLQY_PREDICT = [
    'plqy'
]

TASK_SPECS = {
    'system': {
        'head': 'multiclass', 
        'out_dim': 7,
        'weight': 1, 
    },
    'direct_gap': {
        'head': 'regression', 
        'out_dim': 1,
        'weight': 1, 
    },
    'band_gap': {
        'head': 'regression', 
        'out_dim': 1,
        'weight': 1, 
    },
    'efermi': {
        'head': 'regression', 
        'out_dim': 1,
        'weight': 1, 
    },
    'is_gap_direct': {
        'head': 'binary', 
        'out_dim': 1,
        'weight': 1, 
    },
}

ABS_TASK_SPECS = {
    'integrated_absorption': {
        'head': 'regression', 
        'out_dim': 1,
        'weight': 1, 
    },
    'integrated_absorption_visible': {
        'head': 'regression', 
        'out_dim': 1,
        'weight': 1, 
    },
    'average_absorption_visible': {
        'head': 'regression', 
        'out_dim': 1,
        'weight': 1, 
    },
}

PLQY_TASK_SPECS = {
    'plqy': {
        'head': 'regression', 
        'out_dim': 1, 
        'weight': 1,
    }
}

DIMENSION_CNT = 2


def get_num_cpus(default=1):
    if "SLURM_CPUS_PER_TASK" in os.environ:
        return int(os.environ["SLURM_CPUS_PER_TASK"])
    try:
        return len(os.sched_getaffinity(0))
    except Exception as e:
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


def get_max_dist(data_dir):
    with open(f'{data_dir}/bounds.json', 'r') as f:
        return json.load(f)['max_bond_dist']