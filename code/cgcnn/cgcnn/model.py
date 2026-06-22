from __future__ import print_function, division

import pickle
import torch
import torch.nn as nn
import torchPersLay as tp


class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    """
    def __init__(self, atom_fea_len, nbr_fea_len):
        """
        Initialize ConvLayer.

        Parameters
        ----------

        atom_fea_len: int
          Number of atom hidden features.
        nbr_fea_len: int
          Number of bond features.
        """
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2*self.atom_fea_len+self.nbr_fea_len,
                                 2*self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2*self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors

        Parameters
        ----------

        atom_in_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom hidden features before convolution
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom

        Returns
        -------

        atom_out_fea: nn.Variable shape (N, atom_fea_len)
          Atom hidden features after convolution

        """
        # TODO will there be problems with the index zero padding?
        N, M = nbr_fea_idx.shape
        # convolution
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
             atom_nbr_fea, nbr_fea], dim=2)
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea.view(
            -1, self.atom_fea_len*2)).view(N, M, self.atom_fea_len*2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out
    

class CrystalGraphEncoder(nn.Module):
    def __init__(
            self, orig_atom_fea_len, nbr_fea_len,
            atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
            vec_fea_len=64, n_vec=1,
            # classification=False, num_classes=2,
            vector='none', weight='none', phi='none',
            dims=3, root_dir='data/graph_data'
    ):
        super(CrystalGraphEncoder, self).__init__()

        assert vector in ['none', 'image', 'landscape', 'perslay'], 'incorrect vectorization input!'
        assert weight in ['none', 'power', 'grid', 'gaussian']  # only none or power
        assert phi in ['none', 'image', 'landscape', 'betti']   # only none or image
        
        self.vector = vector
        self.weight = weight
        self.phi = phi

        if self.vector == 'none':
            self.vector_len = 0
        elif self.vector == 'image':
            self.vector_len = 400
        elif self.vector == 'landscape':
            self.vector_len = 500
        elif self.vector == 'perslay':
            if self.phi == 'image':
                self.vector_len = 400

            # ! Don't use landscape/betti for now
            # TentPerslayPhi/FlatPerslayPhi function doesn't compute k-th highest tent,
            # it concatenates tent/betti functions (len: sample) for each point --> sample * point
            # so vectorization is absurdly long

            # elif self.phi == 'landscape':
            #     pass
            # elif self.phi == 'betti':
            #     pass
        
        # self.classification = classification
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                    nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])
        
        # ! vectorization concatenation length
        conv_to_fc_input_len = atom_fea_len + vec_fea_len
        self.conv_to_fc = nn.Linear(conv_to_fc_input_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h-1)])
            self.softpluses = nn.ModuleList([nn.Softplus() for _ in range(n_h-1)])
        
        # ! adapt to multitask
        # if self.classification:
        #     self.fc_out = nn.Linear(h_fea_len, num_classes)  # ! Changed 2 --> 7 (for systems)
        # else:
        #     self.fc_out = nn.Linear(h_fea_len, 1)
        # if self.classification:
        #     self.logsoftmax = nn.LogSoftmax(dim=1)
        #     self.dropout = nn.Dropout()

        # ! vectorization process layers
        self.vec_embedding = nn.Linear(self.vector_len * dims, vec_fea_len * dims)
        self.vec_fcs = nn.ModuleList([nn.Linear(vec_fea_len * dims, vec_fea_len * dims)
                                     for _ in range(n_vec)])
        self.vec_pooling = nn.Linear(vec_fea_len * dims, vec_fea_len)

        # ! perslay --> experiment with parameters
        if self.vector == 'perslay':
            self.weights = nn.ModuleList([
                tp.PowerPerslayWeight(
                    constant=1.0,   # learnable
                    power=1.0
                )
                for _ in range(dims)
            ])
            # input (read from pickle)
            with open(f'{root_dir}/bounds.pkl', 'rb') as file:
                self.image_bnds = pickle.load(file)
            self.phis = nn.ModuleList([
                tp.GaussianPerslayPhi(
                    image_size=(20, 20),
                    image_bnds=self.image_bnds[i],
                    variance=0.1,   # learnable
                )
                for i in range(dims)
            ])
            self.perm_op = torch.sum
            self.rho = nn.Identity()

            self.perslays = nn.ModuleList([
                tp.Perslay(weight=self.weights[i], phi=self.phis[i], perm_op=self.perm_op, rho=self.rho)
                for i in range(dims)
            ])


class MLPHead(nn.Module):
    '''
    Small MLP prediction head for individual crystal properties. 
    Can modify hidden/output dimension, as well as layer depth. 
    '''
    def __init__(self, hidden_dim, output_dim, layer_cnt=1):
        super(MLPHead, self).__init__()
        layers = []
        for _ in range(layer_cnt):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Softplus())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        return self.model(input)


class CrystalGraphConvNet(nn.Module):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    """
    def __init__(
            self, orig_atom_fea_len, nbr_fea_len,
            atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
            vec_fea_len=64, n_vec=1,
            # classification=False, num_classes=2,
            vector='none', weight='none', phi='none',
            dims=3, root_dir='data/graph_data'
    ):
        """
        Initialize CrystalGraphConvNet.

        Parameters
        ----------

        orig_atom_fea_len: int
          Number of atom features in the input.
        nbr_fea_len: int
          Number of bond features.
        atom_fea_len: int
          Number of hidden atom features in the convolutional layers
        n_conv: int
          Number of convolutional layers
        h_fea_len: int
          Number of hidden features after pooling
        n_h: int
          Number of hidden layers after pooling
        """
        super(CrystalGraphConvNet, self).__init__()
        
        assert vector in ['none', 'image', 'landscape', 'perslay'], 'incorrect vectorization input!'
        assert weight in ['none', 'power', 'grid', 'gaussian']  # only none or power
        assert phi in ['none', 'image', 'landscape', 'betti']   # only none or image
        
        self.vector = vector
        self.weight = weight
        self.phi = phi

        if self.vector == 'none':
            self.vector_len = 0
        elif self.vector == 'image':
            self.vector_len = 400
        elif self.vector == 'landscape':
            self.vector_len = 500
        elif self.vector == 'perslay':
            if self.phi == 'image':
                self.vector_len = 400

            # ! Don't use landscape/betti for now
            # TentPerslayPhi/FlatPerslayPhi function doesn't compute k-th highest tent,
            # it concatenates tent/betti functions (len: sample) for each point --> sample * point
            # so vectorization is absurdly long

            # elif self.phi == 'landscape':
            #     pass
            # elif self.phi == 'betti':
            #     pass
        
        # self.classification = classification
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                    nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])
        
        # ! vectorization concatenation length
        conv_to_fc_input_len = atom_fea_len + vec_fea_len
        self.conv_to_fc = nn.Linear(conv_to_fc_input_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h-1)])
            self.softpluses = nn.ModuleList([nn.Softplus() for _ in range(n_h-1)])
        
        # ! multitask
        # if self.classification:
        #     self.fc_out = nn.Linear(h_fea_len, num_classes)  # ! Changed 2 --> 7 (for systems)
        # else:
        #     self.fc_out = nn.Linear(h_fea_len, 1)
        if self.classification:
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()

        # ! vectorization process layers
        self.vec_embedding = nn.Linear(self.vector_len * dims, vec_fea_len * dims)
        self.vec_fcs = nn.ModuleList([nn.Linear(vec_fea_len * dims, vec_fea_len * dims)
                                     for _ in range(n_vec)])
        self.vec_pooling = nn.Linear(vec_fea_len * dims, vec_fea_len)

        # ! perslay --> experiment with parameters
        if self.vector == 'perslay':
            self.weights = nn.ModuleList([
                tp.PowerPerslayWeight(
                    constant=1.0,   # learnable
                    power=1.0
                )
                for _ in range(dims)
            ])
            # input (read from pickle)
            with open(f'{root_dir}/bounds.pkl', 'rb') as file:
                self.image_bnds = pickle.load(file)
            self.phis = nn.ModuleList([
                tp.GaussianPerslayPhi(
                    image_size=(20, 20),
                    image_bnds=self.image_bnds[i],
                    variance=0.1,   # learnable
                )
                for i in range(dims)
            ])
            self.perm_op = torch.sum
            self.rho = nn.Identity()

            self.perslays = nn.ModuleList([
                tp.Perslay(weight=self.weights[i], phi=self.phis[i], perm_op=self.perm_op, rho=self.rho)
                for i in range(dims)
            ])

    def forward(
            self, 
            atom_fea, nbr_fea, nbr_fea_idx, 
            crystal_atom_idx,
            vectorizations, 
            diagrams,    # [d0, d1, d2]
    ):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, orig_atom_fea_len)
          Atom features from atom type
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx

        Returns
        -------

        prediction: nn.Variable shape (N, )
          Atom hidden features after convolution

        """
        ################## START OF ENCODER ##################
        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)

        # ! concatenating vectorizations
        if self.vector != 'none':
            # initialize vec embeddings
            if self.vector in ['image', 'landscape']:
                vec_fea = vectorizations
            elif self.vector == 'perslay':
                vec_fea = torch.cat([
                    self.perslays[i](diagrams[i]).squeeze(-1).flatten(start_dim=1)
                    for i in range(len(self.perslays))
                ], dim=1)
            # vec processing layers
            vec_fea = self.vec_embedding(vectorizations)
            for vec_fc in self.vec_fcs:
                vec_fea = vec_fc(vec_fea)
            vec_fea = self.vec_pooling(vec_fea)
                
            # concatenating processed vec to crystal features
            crys_fea = torch.cat([crys_fea, vec_fea], dim=1)

        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        ################## END OF ENCODER ##################

        if self.classification:
            crys_fea = self.dropout(crys_fea)
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        out = self.fc_out(crys_fea)
        if self.classification:         # ! maybe don't use softmax? let loss handle logits directly?
            out = self.logsoftmax(out)
        return out

    def pooling(self, atom_fea, crystal_atom_idx):
        """
        Pooling the atom features to crystal features

        N: Total number of atoms in the batch
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom feature vectors of the batch
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx
        """
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) ==\
            atom_fea.data.shape[0]
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)
