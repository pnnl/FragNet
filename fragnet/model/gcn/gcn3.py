from torch.nn import Parameter
from torch_geometric.utils import add_self_loops, degree
import torch
import torch.nn as nn
from torch_scatter import scatter_add, scatter_softmax
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter_add


class FragNetLayer(nn.Module):
    """
    gin as implemented in https://github.com/snap-stanford/pretrain-gnns/blob/master/chem/model.py
    """
    def __init__(self, atom_in=128, atom_out=128, frag_in=128, frag_out=128,
                 edge_in=128, edge_out=128):
        super(FragNetLayer, self).__init__()


        self.atom_embed = nn.Linear(atom_in, atom_out, bias=True)
        self.frag_embed = nn.Linear(frag_in, frag_out)
        self.edge_embed = nn.Linear(edge_in, edge_out)
        
        self.frag_message_mlp = nn.Linear(atom_out*2, atom_out)
        self.atom_mlp = torch.nn.Sequential(torch.nn.Linear(atom_out, 2*atom_out),
                                               torch.nn.ReLU(),
                                               torch.nn.Linear(2*atom_out, atom_out))
        
        self.frag_mlp = torch.nn.Sequential(torch.nn.Linear(atom_out, 2*atom_out),
                                               torch.nn.ReLU(),
                                               torch.nn.Linear(2*atom_out, atom_out))
        self.edge_attr_bond_embed = nn.Linear(1, edge_out)
        # self.bias = Parameter(torch.Tensor(atom_out))
        
        
    def forward(self, x_atoms, 
                edge_index,
                edge_attr, 
                frag_index,
                x_frags,
                atom_to_frag_ids,
                node_feautures_bond_graph, 
                edge_index_bonds_graph, 
                edge_attr_bond_graph):
        
        
        # adding self loops to edge features of the bond graph
        edge_index_bonds_graph, _ = add_self_loops(edge_index=edge_index_bonds_graph)
        num_nodes_b = node_feautures_bond_graph.size(0)
        self_loop_attr = 1.5*torch.ones(num_nodes_b, 1, dtype=torch.float)
        edge_attr_bond_graph = torch.cat((edge_attr_bond_graph, self_loop_attr.to(edge_attr)), dim=0)

        target, source = edge_index_bonds_graph # does this include edges between node and itself?
        edge_attr_bond_graph = self.edge_attr_bond_embed(edge_attr_bond_graph)

        # ea_bonds = edge_attr_bond_graph.repeat(self.num_heads,1, 1).permute(1,0,2)
        
        node_feats_b = self.edge_embed(node_feautures_bond_graph)
        # node_feats_b = node_feats_b.view(num_nodes_b, self.num_heads, -1)
        
        source_features = torch.index_select(input=node_feats_b, index=source, dim=0)
        target_features = torch.index_select(input=node_feats_b, index=target, dim=0)
        # message = torch.cat([target_features, edge_attr_bond_graph + source_features], dim=-1)
        message = edge_attr_bond_graph + source_features
        edge_attr = scatter_add(src=message, index=target, dim=0) # node_feats_b_new



        # atom graph
        edge_index, _ = add_self_loops(edge_index=edge_index)
        self_loop_attr = torch.zeros(x_atoms.size(0), 128, dtype=torch.long)
        # self_loop_attr[:,0] = 0 # bond type for self-loop edge
        edge_attr = torch.cat((edge_attr, self_loop_attr.to(edge_attr)), dim=0)
    
    
        x_atoms = self.atom_embed(x_atoms)
        # edge_attr = self.edge_embed(edge_attr)
        source, target = edge_index

        source_features = torch.index_select(input=x_atoms, index=source, dim=0)
        message = edge_attr + source_features
        
        
        # deg = degree(source, x_atoms.size(0), dtype=x_atoms.dtype)
        # deg_inv_sqrt = deg.pow(-0.5)
        # deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        # norm = deg_inv_sqrt[source] * deg_inv_sqrt[target]
        
        # message = source_features*norm.view(-1,1)
        

        x_atoms_new = scatter_add(src=message, index=target, dim=0)
                        
        x_frags  = scatter_add(src=x_atoms_new, index=atom_to_frag_ids, dim=0)
        
        source, target = frag_index
        source_features = torch.index_select(input=x_frags, index=source, dim=0)

        frag_message = source_features
        frag_feats_sum = scatter_add(src=frag_message, index=target, dim=0)
        frag_feats_sum = self.frag_mlp(frag_feats_sum)
        
        x_frags_new = frag_feats_sum
        
        
        return x_atoms_new, x_frags_new


    
    
class FragNet(nn.Module):

    def __init__(self, num_layer, drop_ratio = 0, emb_dim=128, 
                 atom_features=45, frag_features=45, edge_features=12):
        super(FragNet, self).__init__()
        self.num_layer = num_layer
        self.dropout = nn.Dropout(p=drop_ratio)
        self.act = nn.ReLU()


        self.layers = torch.nn.ModuleList()
        self.layers.append(FragNetLayer(atom_in=atom_features, atom_out=emb_dim, frag_in=frag_features, 
                                   frag_out=emb_dim, edge_in=edge_features, edge_out=emb_dim))
        
        for i in range(num_layer-1):
            self.layers.append(FragNetLayer(atom_in=emb_dim, atom_out=emb_dim, frag_in=emb_dim, 
                                   frag_out=emb_dim, edge_in=edge_features, edge_out=emb_dim))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

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
        
        
        
        x_atoms = self.dropout(x_atoms)
        x_frags = self.dropout(x_frags)
        
        x_atoms, x_frags, x_edges = self.layers[0](x_atoms, edge_index, edge_attr, 
                               frag_index, x_frags, atom_to_frag_ids, 
                               node_feautures_bond_graph, edge_index_bonds_graph, edge_attr_bond_graph)
        
        x_atoms, x_frags = self.act(self.dropout(x_atoms)), self.act(self.dropout(x_frags))

        for layer in self.layers[1:]:
            x_atoms, x_frags = layer(x_atoms, edge_index, edge_attr, 
                                       frag_index, x_frags, atom_to_frag_ids,
                                       node_feautures_bond_graph, edge_index_bonds_graph, edge_attr_bond_graph)
            x_atoms, x_frags = self.act(self.dropout(x_atoms)), self.act(self.dropout(x_frags))
        
        return x_atoms, x_frags

    
class FragNetPreTrain(nn.Module):
    
    def __init__(self):
        super(FragNetPreTrain, self).__init__()
        
        self.pretrain = FragNet(num_layer=6, drop_ratio=0.15)
        self.lin1 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 13)
        self.dropout = nn.Dropout(p=0.15)
        self.activation = nn.ReLU()
        # self.lin2 = nn.Linear(128*2, 128)
        
        
        
#         self.egde_pred = nn.Linear(128, 5)
#         self.atom_pred = nn.Linear(128, len(symbols)+1)
#         self.act = nn.Sigmoid()
        
    def forward(self, batch):
        
        x_atoms, x_frags = self.pretrain(batch)
        
        # x_frags_pooled = scatter_add(src=x_frags, index=batch['frag_batch'], dim=0)
        # x_atoms_pooled = scatter_add(src=x_atoms, index=batch['batch'], dim=0)
        
        # cat = torch.cat((x_atoms_pooled, x_frags_pooled), 1)
        x = self.dropout(x_atoms)
        x = self.lin1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.out(x)
        
    
        return x
    
class FragNetFineTune(nn.Module):
    
    def __init__(self, n_classes=1, num_layer=4, drop_ratio=.15, emb_dim=128):
        super(FragNetFineTune, self).__init__()
        
        self.pretrain = FragNet(num_layer=num_layer, drop_ratio=drop_ratio, emb_dim=emb_dim)
        self.lin1 = nn.Linear(emb_dim*2, emb_dim*2)
        
        self.dropout = nn.Dropout(p=0.15)
        self.activation = nn.ReLU()
        # self.n_classes = n_classes
        self.out = nn.Linear(emb_dim*2, n_classes)
        
#         self.egde_pred = nn.Linear(128, 5)
#         self.atom_pred = nn.Linear(128, len(symbols)+1)
#         self.act = nn.Sigmoid()
        
    def forward(self, batch):
        
        x_atoms, x_frags = self.pretrain(batch)
        
        x_frags_pooled = scatter_add(src=x_frags, index=batch['frag_batch'], dim=0)
        x_atoms_pooled = scatter_add(src=x_atoms, index=batch['batch'], dim=0)
        
        cat = torch.cat((x_atoms_pooled, x_frags_pooled), 1)
        x = self.dropout(cat)
        x = self.lin1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.out(x)
        
    
        return x
    


    class FragNetPreTrain(nn.Module):
    
        def __init__(self, num_layer=4, drop_ratio=0.15, num_heads=4, emb_dim=128):
            super(FragNetPreTrain, self).__init__()
            
            # self.pretrain = FragNet(num_layer=num_layer, drop_ratio=drop_ratio)
            self.pretrain = FragNet(num_layer=num_layer, drop_ratio=drop_ratio, num_heads=num_heads, emb_dim=emb_dim)
            self.head = PretrainTask(128, 1)
            
            
        def forward(self, batch):
            
            x_atoms, x_frags, e_edge = self.pretrain(batch)
            bond_length_pred, bond_angle_pred, dihedral_angle_pred =  self.head(x_atoms, e_edge, batch['edge_index'])
            
            return bond_length_pred, bond_angle_pred, dihedral_angle_pred
        