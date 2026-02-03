from torch_scatter import scatter_add
import matplotlib.cm as cm
# highlightatoms
from fragnet.dataset.fragments import FragmentedMol
import torch.nn as nn
import torch
from fragnet.dataset.fragments import get_3Dcoords
import argparse
from omegaconf import OmegaConf
from fragnet.dataset.dataset import load_pickle_dataset
from fragnet.dataset.data import collate_fn
from torch.utils.data import DataLoader
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
# IPythonConsole.molSize=(450,350)
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import Geometry
# rdDepictor.SetPreferCoordGen(True)
# from rdkit import Chem
import pandas as pd
# from rdkit.Chem import Draw
from collections import defaultdict
import matplotlib.cm as cm
# highlightatoms
from fragnet.dataset.fragments import FragmentedMol
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
import io
from PIL import Image
from collections import defaultdict
from rdkit.Chem.Draw import IPythonConsole
# IPythonConsole.drawOptions.addAtomIndices = True
from fragnet.model.gat.gat2 import FragNetLayerA
from fragnet.model.gat.gat2 import FTHead1, FTHead2,FTHead3,FTHead4
from fragnet.model.gat.gat2 import FragNetLayerA
from fragnet.model.gat.gat2 import FragNet

import torch.nn as nn
from torch_scatter import scatter_add
import torch
from fragnet.train.pretrain.pretrain_heads import PretrainTask
# from gat2 import FragNet

class FragNetViz(nn.Module):

    def __init__(self, num_layer, drop_ratio = 0.2, emb_dim=128, 
                 atom_features=167, frag_features=167, edge_features=17, fedge_in=6, fbond_edge_in=6, num_heads=4,
                bond_mask=None):
        super().__init__()
        self.num_layer = num_layer
        self.dropout = nn.Dropout(p=drop_ratio)
        self.act = nn.ReLU()

        self.layers = torch.nn.ModuleList()
        self.layers.append(FragNetLayerA(atom_in=atom_features, atom_out=emb_dim, frag_in=frag_features, 
                                   frag_out=emb_dim, edge_in=edge_features, fedge_in=fedge_in, fbond_edge_in=fbond_edge_in, edge_out=emb_dim, num_heads=num_heads, bond_mask=bond_mask))
        

        for i in range(num_layer-2):
            self.layers.append(FragNetLayerA(atom_in=emb_dim, atom_out=emb_dim, frag_in=emb_dim, 
                                   frag_out=emb_dim, edge_in=emb_dim, edge_out=emb_dim, fedge_in=emb_dim,
                                             fbond_edge_in=fbond_edge_in,
                                    num_heads=num_heads, bond_mask=bond_mask))
            
        self.layers.append(FragNetLayerA(atom_in=emb_dim, atom_out=emb_dim, frag_in=emb_dim, 
                                   frag_out=emb_dim, edge_in=emb_dim, edge_out=emb_dim, fedge_in=emb_dim,
                                         fbond_edge_in=fbond_edge_in,
                                    num_heads=num_heads, bond_mask=bond_mask, return_attentions=True))


    #def forward(self, x, edge_index, edge_attr):
    def forward(self, batch):
        
        
        x_atoms = batch['x_atoms']
        edge_index = batch['edge_index']
        frag_index = batch['frag_index']
        
        x_frags = batch['x_frags']
        edge_attr = batch['edge_attr']
        atom_batch = batch['batch']
        frag_batch = batch['frag_batch']
        atom_to_frag_ids = batch['atom_to_frag_ids']
        
        node_feautures_bond_graph=batch['node_features_bonds']
        edge_index_bonds_graph=batch['edge_index_bonds_graph']
        edge_attr_bond_graph=batch['edge_attr_bonds']


        node_feautures_fbondg = batch['node_features_fbonds']
        edge_index_fbondg = batch['edge_index_fbonds']
        edge_attr_fbondg = batch['edge_attr_fbonds']
        
        

        
        x_atoms = self.dropout(x_atoms)
        x_frags = self.dropout(x_frags)
        
        
        x_atoms, x_frags, edge_features, fedge_features = self.layers[0](x_atoms, edge_index, edge_attr, 
                               frag_index, x_frags, atom_to_frag_ids,
                               node_feautures_bond_graph,edge_index_bonds_graph,edge_attr_bond_graph,
                               node_feautures_fbondg,edge_index_fbondg,edge_attr_fbondg
                               )
        
        
        x_atoms, x_frags = self.act(self.dropout(x_atoms)), self.act(self.dropout(x_frags))
        edge_features = self.act(self.dropout(edge_features))
        fedge_features = self.act(self.dropout(fedge_features))

        # i=1
        for layer in self.layers[1:-1]:
            x_atoms, x_frags, edge_features, fedge_features = layer(x_atoms, edge_index, edge_features, 
                                       frag_index, x_frags, atom_to_frag_ids,
                                      edge_features, edge_index_bonds_graph, edge_attr_bond_graph,
                                      fedge_features, edge_index_fbondg, edge_attr_fbondg
                                      )
            
            
            x_atoms, x_frags = self.act(self.dropout(x_atoms)), self.act(self.dropout(x_frags))
            edge_features = self.act(self.dropout(edge_features))
            fedge_features = self.act(self.dropout(fedge_features))


        
        x_atoms, x_frags, edge_features, fedge_features, summed_attn_weights_atoms, summed_attn_weights_frags, summed_attn_weights_bonds, summed_attn_weights_fbonds = self.layers[-1](x_atoms, edge_index, edge_features, 
                                       frag_index, x_frags, atom_to_frag_ids,
                                      edge_features, edge_index_bonds_graph, edge_attr_bond_graph,
                                      fedge_features, edge_index_fbondg, edge_attr_fbondg
                                      )
            
            
        x_atoms, x_frags = self.act(self.dropout(x_atoms)), self.act(self.dropout(x_frags))
        edge_features = self.act(self.dropout(edge_features))
        fedge_features = self.act(self.dropout(fedge_features))

            
        
        
        
        return x_atoms, x_frags, edge_features, fedge_features, summed_attn_weights_atoms, summed_attn_weights_frags, summed_attn_weights_bonds, summed_attn_weights_fbonds



class FragNetFineTuneViz(nn.Module):
    
    def __init__(self, n_classes=1, atom_features=167, frag_features=167, edge_features=16, 
                 num_layer=4, num_heads=4, drop_ratio=0.15,
                h1=256, h2=256, h3=256, h4=256, act='celu',emb_dim=128, fthead='FTHead3',
                bond_mask=None):
        super().__init__()
        

        self.pretrain = FragNetViz(num_layer=num_layer, drop_ratio=drop_ratio, 
                                num_heads=num_heads, emb_dim=emb_dim,
                                atom_features=atom_features, frag_features=frag_features, edge_features=edge_features,
                                  bond_mask=bond_mask)
        # self.lin1 = nn.Linear(emb_dim*2, h1)
        # self.out = nn.Linear(h1, n_classes)
        # self.dropout = nn.Dropout(p=drop_ratio)
        # self.activation = nn.ReLU()

        # if not fthead:
        #     fthead='FTHead3'

        if fthead == 'FTHead1':
            self.fthead = FTHead1(n_classes=n_classes)
        elif fthead == 'FTHead2':
            print('using FTHead2' )
            self.fthead = FTHead2(n_classes=n_classes)
        elif fthead == 'FTHead3':
            print('using FTHead3' )
            self.fthead = FTHead3(n_classes=n_classes,
                             h1=h1, h2=h2, h3=h3, h4=h4,
                             drop_ratio=drop_ratio, act=act)
            
        elif fthead == 'FTHead4':
            print('using FTHead4' )
            self.fthead = FTHead4(n_classes=n_classes,
                             h1=h1, drop_ratio=drop_ratio, act=act)
        
                    
        
    def forward(self, batch):
        
        x_atoms, x_frags, x_edge, x_fedge, summed_attn_weights_atoms, summed_attn_weights_frags, \
              summed_attn_weights_bonds, summed_attn_weights_fbonds = self.pretrain(batch)
        
        x_frags_pooled = scatter_add(src=x_frags, index=batch['frag_batch'], dim=0)
        x_atoms_pooled = scatter_add(src=x_atoms, index=batch['batch'], dim=0)
        
        cat = torch.cat((x_atoms_pooled, x_frags_pooled), 1)
        # x = self.dropout(cat)
        # x = self.lin1(x)
        # x = self.activation(x)
        # x = self.dropout(x)
        # x = self.out(x)
        x = self.fthead(cat)
        
    
        return x, summed_attn_weights_atoms, summed_attn_weights_frags, \
              summed_attn_weights_bonds, summed_attn_weights_fbonds






class FragNetPreTrainViz(nn.Module):
    
    def __init__(self, num_layer=4, drop_ratio=0.15, num_heads=4, emb_dim=128,
                 atom_features=167, frag_features=167, edge_features=16):
        super(FragNetPreTrainViz, self).__init__()
        
        # self.pretrain = FragNet(num_layer=num_layer, drop_ratio=drop_ratio)
        self.pretrain = FragNetViz(num_layer=num_layer, drop_ratio=drop_ratio, num_heads=num_heads, emb_dim=emb_dim,
                                atom_features=atom_features, frag_features=frag_features, edge_features=edge_features)
        self.head = PretrainTask(128, 1)
        
        
    def forward(self, batch):
        
        # x_atoms, x_frags, e_edge, e_fedge = self.pretrain(batch)

        x_atoms, x_frags, x_edge, x_fedge, summed_attn_weights_atoms, summed_attn_weights_frags, \
              summed_attn_weights_bonds, summed_attn_weights_fbonds = self.pretrain(batch)
        
        bond_length_pred, bond_angle_pred, dihedral_angle_pred, graph_rep =  self.head(x_atoms, x_frags, x_edge, batch)

        # return bond_length_pred, bond_angle_pred, dihedral_angle_pred, graph_rep

        return graph_rep, summed_attn_weights_atoms, summed_attn_weights_frags, \
              summed_attn_weights_bonds, summed_attn_weights_fbonds
