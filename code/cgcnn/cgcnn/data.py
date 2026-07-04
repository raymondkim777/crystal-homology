from __future__ import print_function, division

import csv
import pickle
import functools
import json
import os
import random
import warnings

import numpy as np
import networkx as nx
import torch
import torch.nn as nn
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


def get_fold_indices(dataset, k=5):
    assert k >= 5, f"[Fold Idx] k-value of {k} is too small"
    total_size = len(dataset)
    indices = list(range(total_size))
    k_size, remainder = divmod(total_size, k)
    
    fold_indices = [
        indices[i * k_size + min(i, remainder) : (i + 1) * k_size + min(i + 1, remainder)]
        for i in range(k)
    ]
    return fold_indices


def get_train_val_test_loader_from_folds(
        dataset, folds, test_fold_idx, collate_fn=default_collate,
        batch_size=64, num_workers=1, pin_memory=False, 
        persistent_workers=False,
):
    assert test_fold_idx < len(folds)
    val_fold_idx = (test_fold_idx - 1) % len(folds)
    # concatenate remaining folds to form train indices
    train_indices = [
        idx
        for fold_idx, fold in enumerate(folds)
        if fold_idx not in [val_fold_idx, test_fold_idx]
        for idx in fold
    ]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(folds[val_fold_idx])
    test_sampler = SubsetRandomSampler(folds[test_fold_idx])

    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              collate_fn=collate_fn, pin_memory=pin_memory, 
                              persistent_workers=persistent_workers)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=pin_memory,
                            persistent_workers=persistent_workers)
    test_loader = DataLoader(dataset, batch_size=batch_size,
                                 sampler=test_sampler,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn, pin_memory=pin_memory,
                                 persistent_workers=persistent_workers)
    return train_loader, val_loader, test_loader


def get_train_val_test_loader(dataset, collate_fn=default_collate,
                              batch_size=64, train_ratio=None,
                              val_ratio=0.1, test_ratio=0.1, return_test=False,
                              num_workers=1, pin_memory=False, persistent_workers=False,
                              **kwargs):
    """
    Utility function for dividing a dataset to train, val, test datasets.

    !!! The dataset needs to be shuffled before using the function !!!

    Parameters
    ----------
    dataset: torch.utils.data.Dataset
      The full dataset to be divided.
    collate_fn: torch.utils.data.DataLoader
    batch_size: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    return_test: bool
      Whether to return the test dataset loader. If False, the last test_size
      data will be hidden.
    num_workers: int
    pin_memory: bool
    # ! added persistent_workers to hopefully speed up

    Returns
    -------
    train_loader: torch.utils.data.DataLoader
      DataLoader that random samples the training data.
    val_loader: torch.utils.data.DataLoader
      DataLoader that random samples the validation data.
    (test_loader): torch.utils.data.DataLoader
      DataLoader that random samples the test data, returns if
        return_test=True.
    """
    total_size = len(dataset)
    if kwargs['train_size'] is None:
        if train_ratio is None:
            assert val_ratio + test_ratio < 1
            train_ratio = 1 - val_ratio - test_ratio
            print(f'[Warning] train_ratio is None, using 1 - val_ratio - '
                  f'test_ratio = {train_ratio} as training data.')
        else:
            assert train_ratio + val_ratio + test_ratio <= 1
    indices = list(range(total_size))
    if kwargs['train_size']:
        train_size = kwargs['train_size']
    else:
        train_size = int(train_ratio * total_size)
    if kwargs['test_size']:
        test_size = kwargs['test_size']
    else:
        test_size = int(test_ratio * total_size)
    if kwargs['val_size']:
        valid_size = kwargs['val_size']
    else:
        valid_size = int(val_ratio * total_size)
    train_sampler = SubsetRandomSampler(indices[:train_size])
    val_sampler = SubsetRandomSampler(
        indices[-(valid_size + test_size):-test_size])
    if return_test:
        test_sampler = SubsetRandomSampler(indices[-test_size:])
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              collate_fn=collate_fn, pin_memory=pin_memory, 
                              persistent_workers=persistent_workers)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=pin_memory,
                            persistent_workers=persistent_workers)
    if return_test:
        test_loader = DataLoader(dataset, batch_size=batch_size,
                                 sampler=test_sampler,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn, pin_memory=pin_memory,
                                 persistent_workers=persistent_workers)
    if return_test:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader


def collate_pool(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal
    properties.

    Parameters
    ----------

    dataset_list: list of tuples for each data point.
      (atom_fea, nbr_fea, nbr_fea_idx), 
      vectorizations, [d0, d1, d2], target, cif_id

      atom_fea: torch.Tensor shape (n_i, atom_fea_len)
      nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
      nbr_fea_idx: torch.LongTensor shape (n_i, M)
      vectorizations: torch.LongTensor shape (vec_len*DIM_CNT, ) 
      d0/1/2: torch.LongTensor shape (pt_cnt, 2) --> [[b, d], ...]
      target: torch.Tensor shape (1, )
      cif_id: str or int

    Returns
    -------
    N = sum(n_i); N0 = sum(i)

    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
      Atom features from atom type
    batch_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
      Bond features of each atom's M neighbors
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
      Indices of M neighbors of each atom
    crystal_atom_idx: list of torch.LongTensor of length N
      Mapping from the crystal idx to atom idx (for each atom, stores crystal idx)
    batch_vectorizations: torch.LongTensor shape (N0, vec_len*DIM_CNT)
    batch_d0/1/2: torch.LongTensor shape (N0, pt_cnt, 2)
    targets: torch.Tensor shape (N0, 8) --> multitask (classification & regression)
      Target value for prediction
    batch_cif_ids: list
    """
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx = []
    batch_vectorizations = []
    batch_diagrams = []
    batch_targets, batch_mask = dict(), dict()
    batch_cif_ids = []
    base_idx = 0
    for i, (
        (atom_fea, nbr_fea, nbr_fea_idx), 
        vectorizations, 
        diagrams,
        targets, 
        mask, 
        cif_id)\
            in enumerate(dataset_list):
        n_i = atom_fea.shape[0]  # number of atoms for this crystal
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx+base_idx)
        new_idx = torch.LongTensor(np.arange(n_i)+base_idx)
        crystal_atom_idx.append(new_idx)
        batch_vectorizations.append(vectorizations)
        for i in range(len(diagrams)):
            if len(batch_diagrams) <= i:
                batch_diagrams.append([])
            batch_diagrams[i].append(diagrams[i])
        for key in targets.keys():
            if key not in batch_targets.keys():
                batch_targets[key] = []
            batch_targets[key].append(targets[key])
        # mask
        for key in mask.keys():
            if key not in batch_mask.keys():
                batch_mask[key] = []
            batch_mask[key].append(mask[key])
        batch_cif_ids.append(cif_id)
        base_idx += n_i

    return (torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            crystal_atom_idx),\
        torch.stack(batch_vectorizations, dim=0),\
        [
            torch.stack(batch_diags, dim=0)
            for batch_diags in batch_diagrams
        ],\
        {
            key: torch.cat(batch_targets[key], dim=0)
            for key in batch_targets.keys()
        },\
        {
            key: torch.cat(batch_mask[key], dim=0)
            for key in batch_mask.keys()
        },\
        batch_cif_ids


class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """
    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)


class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Parameters
    ----------

    elem_embedding_file: str
        The path to the .json file
    """
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


class GraphData(Dataset):
    """
    The GraphData dataset is a wrapper for a dataset where the crystal structures
    are stored in the form of NetworkX graphs. The dataset should have the following
    directory structure:

    root_dir
    ├── id_prop.csv
    ├── atom_init.json
    └── graphs
        ├── id0.pkl
        ├── id1.pkl
        ├── ...

    id_prop.csv: a CSV file with two columns. The first column recodes a
    unique ID for each crystal, and the second column recodes the value of
    target property.

    atom_init.json: a JSON file that stores the initialization vector for each
    element.

    ID.pkl: a Pickle file that encodes the crystal structure as an NX graph, 
    where ID is the unique ID for the crystal.

    Parameters
    ----------
    NEED TO CHANGE
    root_dir: str
        The path to the root directory of the dataset
    max_num_nbr: int
        The maximum number of neighbors while constructing the crystal graph
    radius: float
        The cutoff radius for searching neighbors
    dmin: float
        The minimum distance for constructing GaussianDistance
    step: float
        The step size for constructing GaussianDistance
    random_seed: int
        Random seed for shuffling the dataset

    Returns
    -------

    atom_fea: torch.Tensor shape (n_i, atom_fea_len)
    nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
    nbr_fea_idx: torch.LongTensor shape (n_i, M)
    target: torch.Tensor shape (1, )
    cif_id: str or int
    """
    def __init__(
            self, 
            root_dir, 
            max_num_nbr=36, 
            dmin=0, 
            dmax=17,  # 16.719527690689166
            step=0.2,
            random_seed=42,
            vec_source='graph', 
            vector='none',  # 'none', 'image', 'landscape', 'perslay
            dims=3,
            task_specs=None,
    ):
        self.root_dir = root_dir  # cgcnn/data/graph_data
        self.max_num_nbr = max_num_nbr
        self.task_specs = task_specs
        assert os.path.exists(root_dir), 'root_dir does not exist!'
        id_prop_file = os.path.join(self.root_dir, 'id_prop.csv')
        assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [[[row[0]] + [float(item) for item in row[1:]]] for row in reader]
        # ! appending mask data to prop data
        id_mask_file = os.path.join(self.root_dir, 'id_mask.csv')
        assert os.path.exists(id_mask_file), 'id_mask.csv does not exist!'
        with open(id_mask_file) as f:
            reader = csv.reader(f)
            id_mask_data = [[row[0]] + [float(item) for item in row[1:]] for row in reader]
        for i in range(len(self.id_prop_data)):
            assert self.id_prop_data[i][0][0] == id_mask_data[i][0], 'id_prop and id_mask IDs do not match!'
            self.id_prop_data[i].append(id_mask_data[i])
        predict_file = os.path.join(self.root_dir, 'tasks', 'predict.pkl')
        with open(predict_file, 'rb') as f:
            self.predict_list = pickle.load(f)
        assert len(self.id_prop_data[0][0][1:]) == len(self.predict_list), 'prop count does not match predict count!'
        assert len(self.id_prop_data[0][1][1:]) == len(self.predict_list), 'mask count does not match predict count!'
        
        # ! shuffling (before calling get_train_val_test_loader in main)
        random.seed(random_seed)
        random.shuffle(self.id_prop_data)
        atom_init_file = os.path.join(self.root_dir, 'atom_init.json')
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=dmax, step=step)

        # ! vectorization support
        assert vec_source in ['graph', 'point'], 'incorrect vectorization source input!'
        assert vector in ['none', 'image', 'landscape', 'perslay'], 'incorrect vectorization input!'
        self.dim_cnt = dims
        self.vector = vector
        self.vector_dict = dict()
        ch = vec_source[0]
        if self.vector == 'image':
            with open(os.path.join(self.root_dir, 'vecs', f'images_{ch}.pkl'), 'rb') as file:
                self.vector_dict = pickle.load(file)
        elif self.vector == 'landscape':
            with open(os.path.join(self.root_dir, 'vecs', f'landscapes_{ch}.pkl'), 'rb') as file:
                self.vector_dict = pickle.load(file)
        elif self.vector == 'perslay':
            with open(os.path.join(self.root_dir, 'vecs', f'diagrams_{ch}.pkl'), 'rb') as file:
                self.vector_dict = pickle.load(file)  # technically diagrams, not vector
        

    def __len__(self):
        return len(self.id_prop_data)
    

    def __cart_vector(self, coord_start, coord_end, jimage, matrix):
        jimage = np.array(jimage)

        delta_frac = coord_end + jimage - coord_start
        delta_cart = delta_frac @ matrix
        return delta_cart
    

    @functools.lru_cache(maxsize=None)  # Cache computed attributes
    def _load_graph_dict(self, mp_id):
        with open(os.path.join(self.root_dir, "graphs", f"mp-{mp_id}.pkl"), 'rb') as file:
            return pickle.load(file)


    @functools.lru_cache(maxsize=None)  # Cache compiled data dictionaries
    def __getitem__(self, idx):
        mp_id = self.id_prop_data[idx][0][0]
        # ! multitask targets
        target_list = list(self.id_prop_data[idx][0][1:])
        mask_list = list(self.id_prop_data[idx][1][1:])
        # print("target list:", target_list)
        # print("mask list:", mask_list)

        targets, mask = dict(), dict()
        for i in range(len(self.predict_list)):
            task_type = self.task_specs[self.predict_list[i]]['head']
            if task_type in ['binary', 'multiclass']:
                targets[self.predict_list[i]] = torch.tensor([target_list[i]]).long()
            elif task_type == 'regression':
                targets[self.predict_list[i]] = torch.tensor([target_list[i]]).float()
            else:
                raise ValueError(f'[GraphData] unrecognized target task {task_type}')
            mask[self.predict_list[i]] = torch.tensor([mask_list[i]]).bool()

        graph_dict = self._load_graph_dict(mp_id)
        graph = graph_dict['graph']

        ########################
        # graph_dict format:
        # {
        #     graph: <graph (undirected)>, 
        #     system: <system>,
        #     ... (additional properties to be added)
        # }
        ########################

        # atom features (node features)
        if type(graph.nodes[0]['species']) == int:
            feature_list = [
                self.ari.get_atom_fea(graph.nodes[node]['species']) # species number
                for node in graph.nodes
            ]
        # fractional atom site --> add relative amounts of each atom in list
        elif type(graph.nodes[0]['species']) == dict:
            feature_list = [
                np.sum((
                    sp_w * self.ari.get_atom_fea(sp_n) 
                    for sp_n, sp_w in graph.nodes[node]['species'].items()
                ), axis=0)
                    # self.ari.get_atom_fea(graph.nodes[node]['species']) # species number
                    # for node in graph.nodes
                for node in graph.nodes
            ]
        else:
            raise TypeError(f"[DATA Atom Feature] Incorrect node species data type")

        # add fractional coordinates as periodic coordinates
        periodic_coords = [
            np.hstack([
                [np.cos(2 * np.pi * val), np.sin(2 * np.pi * val)]
                for val in graph.nodes[node]['coords']
            ])
            for node in graph.nodes
        ]
        atom_fea = np.hstack((feature_list, periodic_coords))

        # neighbor features (edge attributes)
        nbr_fea_idx, nbr_fea = [], []
        adj_dict = nx.to_dict_of_dicts(graph)
        for u in adj_dict.keys():
            nbr_list = []
            dist = []
            for v in adj_dict[u].keys():
                for k in adj_dict[u][v].keys():
                    # u v k --> node1, node2, key
                    nbr_list.append(v)
                    dist.append(adj_dict[u][v][k]['weight'])
            nbr_fea_idx.append(nbr_list + [0] * (self.max_num_nbr - len(nbr_list)))
            nbr_fea.append(dist + [0] * (self.max_num_nbr - len(nbr_list)))
        
        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
        nbr_fea = self.gdf.expand(nbr_fea)

        # increase nbr_fea_len by 3 to hold cartesian displacement vectors
        padding = ((0, 0), (0, 0), (0, 3))
        nbr_fea = np.pad(nbr_fea, pad_width=padding, mode='constant', constant_values=0)

        # add to_jimage as cartesian displacement vector
        for u in adj_dict.keys():
            cart_vectors = []
            for v in adj_dict[u].keys():
                for k in adj_dict[u][v].keys():
                    to_jimage = np.asarray(adj_dict[u][v][k]['to_jimage'])
                    coord_start = graph.nodes[u]['coords']
                    coord_end = graph.nodes[v]['coords']
                    matrix = graph_dict['lattice_matrix']

                    cart_vector = self.__cart_vector(coord_start, coord_end, to_jimage, matrix)

                    cart_vectors.append(cart_vector)
            cart_vectors = np.asarray(cart_vectors)
            nbr_fea[u, :cart_vectors.shape[0], -3:] = cart_vectors

        # ! vectorization & normalization (optional)
        if self.vector == 'none':
            vectorizations = np.array([])
            diagrams = [torch.Tensor([]) for _ in range(self.dim_cnt)]
        elif self.vector in ['image', 'landscape']:
            vectorizations = np.hstack([self.vector_dict[f'mp-{mp_id}'][dim] for dim in range(self.dim_cnt)])
            # # normalize vectors (optional, ineffective)
            # if np.sum(vectorizations) != 0:
            #     vec_norm = np.linalg.norm(vectorizations)
            #     vectorizations = vectorizations / vec_norm
            diagrams = [torch.Tensor([]) for _ in range(self.dim_cnt)]
        elif self.vector == 'perslay':
            # ! if perslay, then we pass in diagrams (each should be tensor)
            vectorizations = np.array([])
            diagrams = [torch.Tensor(self.vector_dict[f'mp-{mp_id}'][dim]) for dim in range(self.dim_cnt)]  # list of np.ndarrays

        atom_fea = torch.Tensor(atom_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        vectorizations = torch.Tensor(vectorizations)
        return (atom_fea, nbr_fea, nbr_fea_idx), vectorizations, diagrams, targets, mask, mp_id