import os
import json
import networkx as nx
import matplotlib.pyplot as plt


from pymatgen.core import Structure


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
}

ABS_TASK_SPECS = {
    'max_absorption_energy': {
        'head': 'regression', 
        'out_dim': 1,
        'weight': 1, 
    },
    'integrated_absorption': {
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


def plot_nxgraph(graph: nx.DiGraph, save=None) -> None:
    # Draw the graph with labels
    pos = nx.spring_layout(graph)

    nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=800)
    edge_labels = nx.get_edge_attributes(graph, "weight")
    formatted_labels = {edge: f"{weight:.3f}" for edge, weight in edge_labels.items()}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=formatted_labels)
    
    # Display the plot
    if save is None:
        plt.show()
    else:
        plt.savefig(save)


def choose_sp_for_site(site, tie_tol=1e-3):
    """
    returns one Element from possibly disordered pymatgen site. 
    prioritizes cu, halides, c, h if tied. 
    """
    if site.is_ordered:
        return site.specie
    
    priority = ('Cu', 'F', 'Cl', 'Br', 'I', 'C', 'H')
    priority_rank = {sym: i for i, sym in enumerate(priority)}

    species_occ = site.species
    max_occ = max(float(occ) for occ in species_occ.values())
    tied_species = [
        sp for sp, occ in species_occ.items()
        if abs(occ - max_occ) <= tie_tol
    ]

    best_species = min(
        tied_species, 
        key=lambda sp: (
            priority_rank.get(sp, len(priority_rank)),
            getattr(sp, "symbol", str(sp)),
        )
    )
    return best_species


def make_structure_ordered(structure, tie_tol=1e-3):    
    '''don't change original structure'''
    chosen_species = [
        choose_sp_for_site(site, tie_tol=tie_tol)
        for site in structure
    ]
    site_properties = {
        key: list(vals)
        for key, vals in structure.site_properties.items()
    }
    ordered_structure = Structure(
        lattice=structure.lattice,
        species=chosen_species,
        coords=structure.frac_coords,
        coords_are_cartesian=False,
        site_properties=site_properties,
        labels=structure.labels,
        charge=structure.charge,
        properties=getattr(structure, "properties", None),
    )

    if not ordered_structure.is_ordered:
        raise ValueError("[Structure Ordering] Ordering failed")

    return ordered_structure


def get_max_dist(data_dir):
    with open(f'{data_dir}/bounds.json', 'r') as f:
        return json.load(f)['max_bond_dist']