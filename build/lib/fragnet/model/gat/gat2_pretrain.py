import torch.nn as nn
import torch
from .pretrain_heads import PretrainTask
from .gat2 import FragNet
import random

class FragNetPreTrain(nn.Module):
    
    def __init__(self, num_layer=4, drop_ratio=0.15, num_heads=4, emb_dim=128,
                 atom_features=167, frag_features=167, edge_features=16,
                fedge_in=6, 
                            fbond_edge_in=6):
        super(FragNetPreTrain, self).__init__()
        
        self.pretrain = FragNet(num_layer=num_layer, drop_ratio=drop_ratio, num_heads=num_heads, emb_dim=emb_dim,
                                atom_features=atom_features, frag_features=frag_features, edge_features=edge_features,
                                                       fedge_in=fedge_in, 
                            fbond_edge_in=fbond_edge_in,)
        self.head = PretrainTask(128, 1)
        
        
    def forward(self, batch):
        
        x_atoms, x_frags, e_edge, e_fedge = self.pretrain(batch)
        bond_length_pred, bond_angle_pred, dihedral_angle_pred, graph_rep =  self.head(x_atoms, x_frags, e_edge, batch)

        return bond_length_pred, bond_angle_pred, dihedral_angle_pred, graph_rep
    

class FragNetPreTrainMasked(nn.Module):
    
    def __init__(self, num_layer=4, drop_ratio=0.15, num_heads=4, emb_dim=128,
                 atom_features=167, frag_features=167, edge_features=16,
                fedge_in=6, fbond_edge_in=6):
        super(FragNetPreTrainMasked, self).__init__()
        
        self.pretrain = FragNet(num_layer=num_layer, drop_ratio=drop_ratio, num_heads=num_heads, emb_dim=emb_dim,
                                atom_features=atom_features, frag_features=frag_features, edge_features=edge_features,
                               fedge_in=fedge_in, fbond_edge_in=fbond_edge_in)
        self.head = PretrainTask(128, 1)
        
        
    def forward(self, batch):
        
        x_atoms, x_frags, e_edge, e_fedge = self.pretrain(batch)

        with torch.no_grad():
            n_atoms = x_atoms.shape[0]
            unmask_atoms = random.sample(list(range(n_atoms)), int(n_atoms*.85) )
            x_atoms_masked = torch.zeros(x_atoms.size()) + 0.0
            x_atoms_masked[unmask_atoms] = x_atoms[unmask_atoms]


        bond_length_pred, bond_angle_pred, dihedral_angle_pred, graph_rep =  self.head(x_atoms_masked, x_frags, e_edge, batch)

        return bond_length_pred, bond_angle_pred, dihedral_angle_pred, graph_rep