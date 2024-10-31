from torch.nn import Parameter
from torch_geometric.utils import add_self_loops, degree
import torch
import torch.nn as nn
from torch_scatter import scatter_add, scatter_softmax
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter_add
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn.norm import BatchNorm
import random


class FragNetLayerA(nn.Module):
    def __init__(self, atom_in=128, atom_out=128, frag_in=128, frag_out=128,
                 edge_in=128, edge_out=128, fedge_in=128, num_heads=2, bond_edge_in=1, fbond_edge_in=8,
                return_attentions=False,
                 add_frag_self_loops=False):
        super().__init__()

        self.add_frag_self_loops = add_frag_self_loops
        self.return_attentions = return_attentions
        self.edge_out = edge_out
        self.atom_embed = nn.Linear(atom_in, atom_out, bias=True)
        self.frag_embed = nn.Linear(frag_in, frag_out)
        self.edge_embed = nn.Linear(edge_in, edge_out)
        self.bond_edge_embed = nn.Linear(edge_in, edge_out)
        
        self.frag_message_mlp = nn.Linear(atom_out*2, atom_out)
        self.atom_mlp = torch.nn.Sequential(torch.nn.Linear(atom_out, 2*atom_out),
                                               torch.nn.ReLU(),
                                               torch.nn.Linear(2*atom_out, atom_out))
        
        self.frag_mlp = torch.nn.Sequential(torch.nn.Linear(atom_out, 2*atom_out),
                                               torch.nn.ReLU(),
                                               torch.nn.Linear(2*atom_out, atom_out))
        self.bias = Parameter(torch.Tensor(atom_out))
        
        self.leakyrelu = nn.LeakyReLU(.2)
        self.num_heads= num_heads
        self.edge_attr_bond_embed2 = nn.Linear(edge_out, edge_out)

        edge_out = edge_out//self.num_heads
        self.projection_b = nn.Linear(edge_in, edge_out * self.num_heads, bias=True)
        self.projection_fb = nn.Linear(fedge_in, edge_out * self.num_heads, bias=True)
         

        self.edge_attr_bond_embed = nn.Linear(bond_edge_in, edge_out)
        self.edge_attr_fbond_embed = nn.Linear(fbond_edge_in, edge_out)

        atom_out = atom_out//self.num_heads
        self.projection_a = nn.Linear(atom_in, atom_out * self.num_heads) # NOTE: check this
        self.a_b = nn.Parameter(torch.Tensor(self.num_heads, 2 * edge_out + edge_out )) # One per head
        self.a = nn.Parameter(torch.Tensor(self.num_heads, 2 * atom_out + edge_out*self.num_heads ))
        self.f = nn.Parameter(torch.Tensor(self.num_heads, 2 * atom_out + edge_out*self.num_heads )) # replace this with embed_dim
        self.f_a_b = nn.Parameter(torch.Tensor(self.num_heads, 2 * edge_out + edge_out )) # replace this with embed_dim
        
        nn.init.xavier_uniform_(self.projection_b.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_b.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        nn.init.xavier_uniform_(self.f.data, gain=1.414)
        nn.init.xavier_uniform_(self.f_a_b.data, gain=1.414)
        
        
    def forward(self, x_atoms, 
                edge_index,
                edge_attr, 
                frag_index,
                x_frags,
                atom_to_frag_ids,
                node_feautures_bond_graph,
                edge_index_bonds_graph,
                edge_attr_bond_graph,
                node_feautures_fbond_graph,
                edge_index_fbond_graph,
                edge_attr_fbond_graph):
        
        # new bond features from bond graph
        target, source = edge_index_bonds_graph
        edge_attr_bond_graph = self.edge_attr_bond_embed(edge_attr_bond_graph)
        ea_bonds = edge_attr_bond_graph.repeat(self.num_heads,1, 1).permute(1,0,2)
        num_nodes_b = node_feautures_bond_graph.size(0)
        node_feats_b = self.projection_b(node_feautures_bond_graph)
        node_feats_b = node_feats_b.view(num_nodes_b, self.num_heads, -1)
        
        source_features = torch.index_select(input=node_feats_b, index=source, dim=0)
        target_features = torch.index_select(input=node_feats_b, index=target, dim=0)
        message = torch.cat([target_features, ea_bonds, source_features], dim=-1)          
        
        attn_logits = torch.sum(message * self.a_b, dim=2)
        attn_logits = self.leakyrelu(attn_logits)

        attn_probs = scatter_softmax(attn_logits, target, dim=0) # eij
        hj = torch.index_select(input=node_feats_b, index=source, dim=0) # select hj

        node_feats_b = attn_probs[..., None]*hj # multipy the attention coefficients corresponding to (neighbors - target atom) with the attention 
                                                # coefficients corresponding to the neighbors. [since we are considering edge features when finding the
                                                # attention coefficients, should we use something like attn_probs[..., None]*(hj+ea_bonds) or 

        node_feats_sum_b = scatter_add(src=node_feats_b, index=target, dim=0) # get the sum of neighboring node features
        summed_attn_weights_bonds = scatter_add(attn_probs, source, dim=0) # sum of attention weights attributable to the target atom
        
        new_bond_features = node_feats_sum_b.view(num_nodes_b, -1)
        
        # new bond features from bond graph
        edge_index, _ = add_self_loops(edge_index=edge_index)

        self_loop_attr = torch.zeros(x_atoms.size(0), self.edge_out, dtype=torch.long)
        edge_attr = torch.cat((new_bond_features, self_loop_attr.to(edge_attr)), dim=0) # <- new
    
        source, target = edge_index

        node_features_atom_graph = self.projection_a(x_atoms)
        num_nodes_a = node_features_atom_graph.size(0)

        node_features_atom_graph = node_features_atom_graph.view(num_nodes_a, self.num_heads, -1)


        source_features = torch.index_select(input=node_features_atom_graph, index=source, dim=0) # here, source and target are for the edge_index with self loops added above
        target_features = torch.index_select(input=node_features_atom_graph, index=target, dim=0) # so, in the message below, we can use the new edge attr we found above

        edge_attr_repeat = edge_attr.repeat(self.num_heads,1, 1).permute(1,0,2)
        message = torch.cat([target_features, edge_attr_repeat, source_features], dim=-1)  

        attn_logits = torch.sum(message * self.a, dim=2) # NOTE: replace a by self.a
        attn_logits = self.leakyrelu(attn_logits)
        attn_probs = scatter_softmax(attn_logits, target, dim=0)
        hj = torch.index_select(input=node_features_atom_graph, index=source, dim=0)

        node_features_atom_graph = attn_probs[..., None]*hj # multiply the node features by the attention weights
        node_feats_sum_a = scatter_add(src=node_features_atom_graph, index=target, dim=0) # nodes in the bond graph are the edges in the atom graph
        summed_attn_weights_atoms = scatter_add(attn_probs, source, dim=0) # total attention attributed to each node in the bond graph, or each edge in the atom graph 
        x_atoms_new = node_feats_sum_a.view(num_nodes_a, -1) 
        x_frags  = scatter_add(src=x_atoms_new, index=atom_to_frag_ids, dim=0)


        # get frag bond features.
        target, source = edge_index_fbond_graph

        edge_attr_fbond_graph = self.edge_attr_fbond_embed(edge_attr_fbond_graph)
        ea_fbonds = edge_attr_fbond_graph.repeat(self.num_heads,1, 1).permute(1,0,2)
        num_nodes_fb = node_feautures_fbond_graph.size(0)
        node_feats_fb = self.projection_fb(node_feautures_fbond_graph)
        node_feats_fb = node_feats_fb.view(num_nodes_fb, self.num_heads, -1)
        
        source_features = torch.index_select(input=node_feats_fb, index=source, dim=0)
        target_features = torch.index_select(input=node_feats_fb, index=target, dim=0)
        message = torch.cat([target_features, ea_fbonds, source_features], dim=-1)  
        
        attn_logits = torch.sum(message * self.f_a_b, dim=2)
        attn_logits = self.leakyrelu(attn_logits)

        attn_probs = scatter_softmax(attn_logits, target, dim=0) # eij
        hj = torch.index_select(input=node_feats_fb, index=source, dim=0) # select hj

        node_feats_fb = attn_probs[..., None]*hj # multipy the attention coefficients corresponding to (neighbors - target atom) with the attention 
                                                # coefficients corresponding to the neighbors. [since we are considering edge features when finding the
                                                # attention coefficients, should we use something like attn_probs[..., None]*(hj+ea_bonds) or 
        node_feats_sum_fb = scatter_add(src=node_feats_fb, index=target, dim=0) # get the sum of neighboring node features
        summed_attn_weights_fbonds = scatter_add(attn_probs, source, dim=0) # sum of attention weights attributable to the target atom
        
        new_fbond_features = node_feats_sum_fb.view(num_nodes_fb, -1)

        # get frag bond features
        edge_attr_fbond_new = new_fbond_features

        source, target = frag_index
        num_nodes_f = x_frags.size(0)
        node_features_frag_graph = x_frags.view(num_nodes_f, self.num_heads, -1)
        source_features = torch.index_select(input=node_features_frag_graph, index=source, dim=0) # here, source and target are for the edge_index with self loops added above
        target_features = torch.index_select(input=node_features_frag_graph, index=target, dim=0) # so, in the message below, we can use the new edge attr we found above

        edge_attr_fbond_repeat = edge_attr_fbond_new.repeat(self.num_heads,1, 1).permute(1,0,2)
        message = torch.cat([target_features, edge_attr_fbond_repeat, source_features], dim=-1)  
    
        attn_logits = torch.sum(message * self.f, dim=2)
        attn_logits = self.leakyrelu(attn_logits)

        attn_probs = scatter_softmax(attn_logits, target, dim=0)
        hj = torch.index_select(input=node_features_frag_graph, index=source, dim=0)

        node_features_frag_graph = attn_probs[..., None]*hj # multiply the node features by the attention weights
        node_feats_sum_f = scatter_add(src=node_features_frag_graph, index=target, dim=0) # nodes in the bond graph are the edges in the atom graph
        summed_attn_weights_frags = scatter_add(attn_probs, source, dim=0) # total attention attributed to each node in the frag graph, or each edge in the frag graph 

        x_frags_new = node_feats_sum_f.view(num_nodes_f, -1) 

        if self.return_attentions:
            return x_atoms_new, x_frags_new, new_bond_features, new_fbond_features, summed_attn_weights_atoms, summed_attn_weights_frags, summed_attn_weights_bonds, summed_attn_weights_fbonds
        else:
            return x_atoms_new, x_frags_new, new_bond_features, new_fbond_features



class FragNet(nn.Module):

    def __init__(self, num_layer, drop_ratio = 0.2, emb_dim=128, 
                 atom_features=167, frag_features=167, edge_features=16, fedge_in=6, fbond_edge_in=6, num_heads=4):
        super().__init__()
        self.num_layer = num_layer
        self.dropout = nn.Dropout(p=drop_ratio)
        self.act = nn.ReLU()
        self.layers = torch.nn.ModuleList()
        self.layers.append(FragNetLayerA(atom_in=atom_features, atom_out=emb_dim, frag_in=frag_features, 
                                   frag_out=emb_dim, edge_in=edge_features, fedge_in=fedge_in, fbond_edge_in=fbond_edge_in, edge_out=emb_dim, num_heads=num_heads))
        
        for i in range(num_layer-1):
            self.layers.append(FragNetLayerA(atom_in=emb_dim, atom_out=emb_dim, frag_in=emb_dim, 
                                   frag_out=emb_dim, edge_in=emb_dim, edge_out=emb_dim, fedge_in=emb_dim,
                                             fbond_edge_in=fbond_edge_in,
                                    num_heads=num_heads))

    def forward(self, batch):
        
        
        x_atoms = batch['x_atoms']
        edge_index = batch['edge_index']
        frag_index = batch['frag_index']
        x_frags = batch['x_frags']
        edge_attr = batch['edge_attr']
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

        for layer in self.layers[1:]:
            x_atoms, x_frags, edge_features, fedge_features = layer(x_atoms, edge_index, edge_features, 
                                       frag_index, x_frags, atom_to_frag_ids,
                                      edge_features, edge_index_bonds_graph, edge_attr_bond_graph,
                                      fedge_features, edge_index_fbondg, edge_attr_fbondg
                                      )
            
            x_atoms, x_frags = self.act(self.dropout(x_atoms)), self.act(self.dropout(x_frags))
            edge_features = self.act(self.dropout(edge_features))
            fedge_features = self.act(self.dropout(fedge_features))
        
        return x_atoms, x_frags, edge_features, fedge_features

    
    
    
class FragNetViz(FragNet):
        def __init__(self, num_layer, drop_ratio = 0, emb_dim=128, 
                     atom_features=45, frag_features=45, edge_features=12, num_heads=4,
                     return_attentions=True):
            super().__init__(num_layer=num_layer, drop_ratio = drop_ratio)


            self.num_layer = num_layer
            
            self.layers[num_layer-1] = FragNetLayerA(atom_in=emb_dim, atom_out=emb_dim, frag_in=emb_dim, 
                           frag_out=emb_dim, edge_in=emb_dim, edge_out=emb_dim, num_heads=num_heads, return_attentions=return_attentions)
            self.num_layer = num_layer

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


            x_atoms = self.dropout(x_atoms)
            x_frags = self.dropout(x_frags)


            x_atoms, x_frags, edge_features = self.layers[0](x_atoms, edge_index, edge_attr, 
                                   frag_index, x_frags, atom_to_frag_ids,
                                   node_feautures_bond_graph,edge_index_bonds_graph,edge_attr_bond_graph)

            x_atoms, x_frags = self.act(self.dropout(x_atoms)), self.act(self.dropout(x_frags))
            for layer in self.layers[1:-1]:
                x_atoms, x_frags, edge_features = layer(x_atoms, edge_index, edge_features, 
                                           frag_index, x_frags, atom_to_frag_ids,
                                          edge_features,edge_index_bonds_graph,edge_attr_bond_graph)

                x_atoms, x_frags = self.act(self.dropout(x_atoms)), self.act(self.dropout(x_frags))
            
            x_atoms, x_frags, edge_features, attn_atom, attn_frag, attn_bond = self.layers[-1](x_atoms, edge_index, edge_features, 
                                           frag_index, x_frags, atom_to_frag_ids,
                                          edge_features,edge_index_bonds_graph,edge_attr_bond_graph)

            x_atoms, x_frags = self.act(self.dropout(x_atoms)), self.act(self.dropout(x_frags))
            
            
            
            return x_atoms, x_frags, edge_features, attn_atom, attn_frag, attn_bond
            
    
    
# from fragnet.model.gat.pretrain_heads import PretrainTask
# class FragNetPreTrain2(nn.Module):
    
#     def __init__(self, num_layer=4, drop_ratio=0.15, num_heads=4, emb_dim=128):
#         super(FragNetPreTrain, self).__init__()
        
#         self.pretrain = FragNet(num_layer=num_layer, drop_ratio=drop_ratio, num_heads=num_heads, emb_dim=emb_dim)
#         self.head = PretrainTask(128, 1)
        
        
#     def forward(self, batch):
        
#         x_atoms, x_frags, e_edge = self.pretrain(batch)
#         bond_length_pred, bond_angle_pred, dihedral_angle_pred =  self.head(x_atoms, e_edge, batch['edge_index'])
#         # i am using all the bond angles as node features in the bond graph. so, it's probably not very effective to use 3D-PDT's
#         # bond angle sum as a pretraining target. the model might not learn anything.

#         return bond_length_pred, dihedral_angle_pred

class FTHead1(nn.Sequential):
    def __init__(self, emb_dim=128, h1=128, drop_ratio=.2, n_classes=1):
        super().__init__()


        self.lin1 = nn.Linear(emb_dim*2, h1)
        self.out = nn.Linear(h1, n_classes)
        self.dropout = nn.Dropout(p=drop_ratio)
        self.activation = nn.ReLU()

    def forward(self, enc):
  
        x = self.dropout(enc)
        x = self.lin1(x)
        x = self.activation(x)
        x = self.dropout(x)
        out = self.out(x)

        return out   



class FTHead5(nn.Sequential):
    def __init__(self, input_dim=128, h1=128, h2=1024, 
                 h4=512, drop_ratio=.2, n_classes=1, act='relu'):
        super().__init__()

        self.dropout = nn.Dropout(p=drop_ratio)
        if act == 'relu':
            self.activation = nn.ReLU()
        elif act == 'silu':
            self.activation = nn.SiLU()
        elif act == 'gelu':
            self.activation = nn.GELU()
        elif act == 'celu':
            self.activation = nn.CELU()
        elif act == 'selu':
            self.activation = nn.SELU()
        elif act == 'rrelu':
            self.activation = nn.RReLU()
        elif act == 'relu6':
            self.activation = nn.ReLU6()
        elif act == 'prelu':
            self.activation = nn.PReLU()
        elif act == 'leakyrelu':
            self.activation = nn.LeakyReLU()
            
        self.hidden_dims =  [h1, h2]
        layer_size = len(self.hidden_dims) + 1
        dims = [input_dim*2] + self.hidden_dims + [n_classes]
        self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(layer_size)])
    

    def forward(self, enc):

        for i in range(0, len(self.predictor)-1):
            enc = self.activation(self.dropout(self.predictor[i](enc)))
        out = self.predictor[-1](enc)

        return out

class FTHead4(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim=128, h1=128,  act='relu',
        n_classes=1,
        drop_ratio=.2
    ):
        super().__init__()

        if act == 'relu':
            self.activation = nn.ReLU()
        elif act == 'gelu':
            self.activation = nn.GELU()
        elif act == 'silu':
            self.activation = nn.SiLU()
        elif act == 'celu':
            self.activation = nn.CELU()
        elif act == 'selu':
            self.activation = nn.SELU()
        elif act == 'rrelu':
            self.activation = nn.RReLU()
        elif act == 'relu6':
            self.activation = nn.ReLU6()
        elif act == 'prelu':
            self.activation = nn.PReLU()
        elif act == 'leakyrelu':
            self.activation = nn.LeakyReLU()

        self.dense = nn.Linear(input_dim*2, h1)
        self.dropout = nn.Dropout(p=drop_ratio)
        self.out_proj = nn.Linear(h1, n_classes)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class FTHead3(nn.Sequential):
    def __init__(self, input_dim=128, h1=128, h2=1024, h3=1024, 
                 h4=512, drop_ratio=.2, n_classes=1, act='relu'):
        super().__init__()


        self.dropout = nn.Dropout(p=drop_ratio)
        if act == 'relu':
            self.activation = nn.ReLU()
        elif act == 'silu':
            self.activation = nn.SiLU()
        elif act == 'gelu':
            self.activation = nn.GELU()
        elif act == 'celu':
            self.activation = nn.CELU()
        elif act == 'selu':
            self.activation = nn.SELU()
        elif act == 'rrelu':
            self.activation = nn.RReLU()
        elif act == 'relu6':
            self.activation = nn.ReLU6()
        elif act == 'prelu':
            self.activation = nn.PReLU()
        elif act == 'leakyrelu':
            self.activation = nn.LeakyReLU()
            
        self.hidden_dims =  [h1, h2, h3, h4]
        layer_size = len(self.hidden_dims) + 1
        dims = [input_dim*2] + self.hidden_dims + [n_classes]
        self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(layer_size)])
    

    def forward(self, enc):
  
        for i in range(0, len(self.predictor)-1):
            enc = self.activation(self.dropout(self.predictor[i](enc)))
        out = self.predictor[-1](enc)

        return out
    


class FTHead2(nn.Sequential):
    def __init__(self, input_dim=128, h1=128, drop_ratio=.2, n_classes=1):
        super().__init__()

        self.lin1 = nn.Linear(input_dim*2, h1)
        self.out = nn.Linear(h1, n_classes)
        self.dropout = nn.Dropout(p=drop_ratio)
        self.activation = nn.ReLU()
  
        self.hidden_dims =  [1024, 1024, 512] #best
        layer_size = len(self.hidden_dims) + 1
        dims = [input_dim*2] + self.hidden_dims + [n_classes]
        self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(layer_size)])
        self.dropout = nn.Dropout(p=.1)

    def forward(self, enc):
  
        for i in range(0, len(self.predictor)-1):
            enc = F.relu(self.dropout(self.predictor[i](enc)))
        out = self.predictor[-1](enc)

        return out   

def do_nothing(a):
    pass

class FragNetFineTune(nn.Module):
    
    def __init__(self, n_classes=1, atom_features=167, frag_features=167, edge_features=16, 
                 num_layer=4, num_heads=4, drop_ratio=0.15,
                h1=256, h2=256, h3=256, h4=256, act='celu',emb_dim=128, fthead='FTHead3'):
        super().__init__()
        

        self.pretrain = FragNet(num_layer=num_layer, drop_ratio=drop_ratio, 
                                num_heads=num_heads, emb_dim=emb_dim,
                                atom_features=atom_features, frag_features=frag_features, edge_features=edge_features)

        if fthead == 'FTHead1':
            self.fthead = FTHead1(n_classes=n_classes)
        elif fthead == 'FTHead2':
            print('using FTHead2' )
            self.fthead = FTHead2(n_classes=n_classes)
        elif fthead == 'FTHead3':
            print('using FTHead3' )
            self.fthead = FTHead3(n_classes=n_classes,
                                  input_dim=emb_dim,
                             h1=h1, h2=h2, h3=h3, h4=h4,
                             drop_ratio=drop_ratio, act=act)
            
        elif fthead == 'FTHead4':
            print('using FTHead4' )
            self.fthead = FTHead4(n_classes=n_classes,
                             h1=h1, drop_ratio=drop_ratio, act=act)
        
                    
        
    def forward(self, batch):
        
        x_atoms, x_frags, x_edge, x_fedge = self.pretrain(batch)

        x_frags_pooled = scatter_add(src=x_frags, index=batch['frag_batch'], dim=0)
        x_atoms_pooled = scatter_add(src=x_atoms, index=batch['batch'], dim=0)
        
        cat = torch.cat((x_atoms_pooled, x_frags_pooled), 1)
        x = self.fthead(cat)
        
        return x
    

from torch_geometric.nn import TransformerConv
class FragNetFineTuneTransformer(nn.Module):
    
    def __init__(self, n_classes=1, num_layer=4, drop_ratio=0.15,
                h1=256, num_heads=4, emb_dim=128, transformer_heads=1,
                atom_features=45, frag_features=45, edge_features=12):
        super().__init__()
        

        self.pretrain = FragNet(num_layer=num_layer, drop_ratio=drop_ratio, 
                                num_heads=num_heads, emb_dim=emb_dim, 
                                atom_features=atom_features, frag_features=frag_features, edge_features=edge_features)
        self.lin1 = nn.Linear(emb_dim*2, h1)
        self.out = nn.Linear(h1, n_classes)
        self.dropout = nn.Dropout(p=drop_ratio)
        self.activation = nn.ReLU()
        
        self.atom_transformer =  TransformerConv(in_channels=emb_dim, out_channels=emb_dim, heads=transformer_heads)
        self.frag_transformer =  TransformerConv(in_channels=emb_dim, out_channels=emb_dim, heads=transformer_heads)
        
                
            
        
    def forward(self, batch):
        
        x_atoms, x_frags, x_edge = self.pretrain(batch)
        
        x_atoms = self.atom_transformer(x=x_atoms, edge_index=batch['edge_index'])
        x_frags = self.atom_transformer(x=x_frags, edge_index=batch['frag_index'])
        
        x_frags_pooled = scatter_add(src=x_frags, index=batch['frag_batch'], dim=0)
        x_atoms_pooled = scatter_add(src=x_atoms, index=batch['batch'], dim=0)
        
        cat = torch.cat((x_atoms_pooled, x_frags_pooled), 1)
        x = self.dropout(cat)
        x = self.lin1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.out(x)
        
    
        return x
    
    
    
    

class FragNetFineTuneMultiTask(FragNetFineTune):
    
    def __init__(self, n_classes=1, num_layer=4, drop_ratio=0.15, 
                 n_multi_task_heads=0):
        super().__init__(n_classes=n_classes, num_layer=num_layer, drop_ratio=drop_ratio)
        
        if n_multi_task_heads>0:
            self.ms_heads = nn.ModuleList([nn.Linear(128*2, n_classes) for i in range(n_multi_task_heads)])
                
            
        
    def forward(self, batch):
        
        x_atoms, x_frags = self.pretrain(batch)
        
        x_frags_pooled = scatter_add(src=x_frags, index=batch['frag_batch'], dim=0)
        x_atoms_pooled = scatter_add(src=x_atoms, index=batch['batch'], dim=0)
        
        cat = torch.cat((x_atoms_pooled, x_frags_pooled), 1)
        x = self.dropout(cat)
        x = self.lin1(x)
        x = self.activation(x)
        x = self.dropout(x)
        # x = self.out(x)
        
        outs = []
        for layer in self.ms_heads:
            outs.append(layer(x))
        
        return outs
    
    
    
class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim # 128
        self.num_heads = num_heads # 8
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, batch_ids=None, return_attention=False, attn_bias=None):
        
        
        qkv = self.qkv_proj(x)
        # batch_ids = batch['batch']
        values1, counts = np.unique(batch_ids.cpu(), return_counts=True)
        batch_size = len(values1)
        seq_length = max(counts)

        a = torch.split(qkv,  list(counts))
        fin = pad_sequence(a, batch_first=True, padding_value = 1)

        mask = fin.sum(-1).ge(fin.shape[-1])
        mask = mask.unsqueeze(1).unsqueeze(2).to(torch.bool)

        qkv = fin.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits * ( d_k**(-0.5) )
        attn_logits = attn_logits.masked_fill(mask, float('-inf') )

        attention = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v)

        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        values = values.reshape(batch_size*seq_length, self.embed_dim)
        m = torch.where(~mask.flatten())
        values = torch.index_select(input=values, dim = 0, index=m[0])
 
        
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o
        
        
class EncoderBlock(nn.Module):

    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim)
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, batch_ids, attn_bias=None):
        # Attention part
        attn_out = self.self_attn(x, batch_ids, attn_bias=attn_bias)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x
    
    
class TransformerEncoder(nn.Module):

    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x, batch_ids,  attn_bias=None):
        for l in self.layers:
            x = l(x, batch_ids=batch_ids, attn_bias=attn_bias)
        return x

        
from torch.nn.utils.rnn import pad_sequence
class FragNetFineTuneTransformer2(nn.Module):
    
    def __init__(self, n_classes=1, num_layer=4, drop_ratio=0.15,
                h1=256, num_heads=4, emb_dim=128, transformer_heads=1,
                num_attn_layer2=6, num_attn_heads2=4, drop_ratio2=.3):
        super().__init__()
        

        self.pretrain = FragNet(num_layer=num_layer, drop_ratio=drop_ratio, 
                                num_heads=num_heads, emb_dim=emb_dim)
        self.lin1 = nn.Linear(emb_dim*2, h1)
        self.out = nn.Linear(h1, n_classes)
        self.dropout = nn.Dropout(p=drop_ratio)
        self.activation = nn.ReLU()
        
        self.transformer = TransformerEncoder(num_layers=num_attn_layer2,
                                      input_dim=emb_dim,
                                      dim_feedforward=2*emb_dim,
                                      num_heads=num_attn_heads2,
                                      dropout=drop_ratio2)
        self.transformer2 = TransformerEncoder(num_layers=num_attn_layer2,
                                      input_dim=emb_dim,
                                      dim_feedforward=2*emb_dim,
                                      num_heads=num_attn_heads2,
                                      dropout=drop_ratio2)
            
            
    
    def forward(self, batch):
        
        x_atoms, x_frags = self.pretrain(batch)
        
        
        x_atoms = self.transformer(x_atoms, batch['batch'])
        x_frags = self.transformer2(x_frags, batch['frag_batch'])
            
        x_frags_pooled = scatter_add(src=x_frags, index=batch['frag_batch'], dim=0)
        x_atoms_pooled = scatter_add(src=x_atoms, index=batch['batch'], dim=0)
        
        cat = torch.cat((x_atoms_pooled, x_frags_pooled), 1)
        x = self.dropout(cat)
        x = self.lin1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.out(x)
        
    
        return x
