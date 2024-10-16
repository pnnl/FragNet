from torch.nn import Parameter
from torch_geometric.utils import add_self_loops, degree
import torch
import torch.nn as nn
from torch_scatter import scatter_add, scatter_softmax
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter_add


class FragNetLayer(nn.Module):
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
        self.bias = Parameter(torch.Tensor(atom_out))
        
        self.leakyrelu = nn.LeakyReLU(.2)
        self.num_heads=3
        edge_out = edge_in//self.num_heads
        self.projection_b = nn.Linear(edge_in, edge_out * self.num_heads, bias=True)
        self.a_b = nn.Parameter(torch.Tensor(self.num_heads, 2 * edge_out + 1 )) # One per head
        
        # nn.init.xavier_uniform_(self.projection.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.projection_b.weight.data, gain=1.414)
        # nn.init.xavier_uniform_(self.a.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_b.data, gain=1.414)
        
        
    def forward(self, x_atoms, 
                edge_index,
                edge_attr, 
                frag_index,
                x_frags,
                atom_to_frag_ids,
                node_feautures_bond_graph,
                edge_index_bonds_graph,
                edge_attr_bond_graph):
        
        
        
        # new bond features from bond graph
        target, source = edge_index_bonds_graph
        
        ea_bonds = edge_attr_bond_graph.repeat(self.num_heads,1, 1).permute(1,0,2)
        num_nodes_b = node_feautures_bond_graph.size(0)
        node_feats_b = self.projection_b(node_feautures_bond_graph)
        node_feats_b = node_feats_b.view(num_nodes_b, self.num_heads, -1)
        
        source_features = torch.index_select(input=node_feats_b, index=source, dim=0)
        target_features = torch.index_select(input=node_feats_b, index=target, dim=0)
        message = torch.cat([target_features, ea_bonds, source_features], dim=-1)  
        
        
        attn_logits = torch.sum(message * self.a_b, dim=2)
        attn_logits = self.leakyrelu(attn_logits)

        attn_probs = scatter_softmax(attn_logits, target, dim=0)
        hj = torch.index_select(input=node_feats_b, index=source, dim=0)

        node_feats_b = attn_probs[..., None]*hj
        node_feats_sum_b = scatter_add(src=node_feats_b, index=target, dim=0)
        summed_attn_weights_bonds = scatter_add(attn_probs, source, dim=0)
        
        new_bond_features = node_feats_sum_b.view(num_nodes_b, -1)
        
        # new bond features from bond graph
        edge_index, _ = add_self_loops(edge_index=edge_index)

        self_loop_attr = torch.zeros(x_atoms.size(0), 12, dtype=torch.long)
        self_loop_attr[:,0] = 0 # bond type for self-loop edge
        edge_attr = torch.cat((new_bond_features, self_loop_attr.to(edge_attr)), dim=0) # <- new
    
        x_atoms = self.atom_embed(x_atoms)
        edge_attr = self.edge_embed(edge_attr)
        source, target = edge_index
        source_features = torch.index_select(input=x_atoms, index=source, dim=0)

        
        deg = degree(source, x_atoms.size(0), dtype=x_atoms.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[source] * deg_inv_sqrt[target]
        
        message = source_features*norm.view(-1,1)
        
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


        self.layer1 = FragNetLayer(atom_in=atom_features, atom_out=emb_dim, frag_in=frag_features, 
                                   frag_out=emb_dim, edge_in=edge_features, edge_out=emb_dim)
        self.layer2 = FragNetLayer(atom_in=emb_dim, atom_out=emb_dim, frag_in=emb_dim, 
                                   frag_out=emb_dim, edge_in=edge_features, edge_out=emb_dim)
        self.layer3 = FragNetLayer(atom_in=emb_dim, atom_out=emb_dim, frag_in=emb_dim, 
                                   frag_out=emb_dim, edge_in=edge_features, edge_out=emb_dim)
        self.layer4 = FragNetLayer(atom_in=emb_dim, atom_out=emb_dim, frag_in=emb_dim, 
                                   frag_out=emb_dim, edge_in=edge_features, edge_out=emb_dim)

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

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
        
        x_atoms, x_frags = self.layer1(x_atoms, edge_index, edge_attr, 
                                       frag_index, x_frags, atom_to_frag_ids,
                                       node_feautures_bond_graph,edge_index_bonds_graph,edge_attr_bond_graph)
        x_atoms, x_frags = self.act(x_atoms), self.act(x_frags)
        
        x_atoms, x_frags = self.layer2(x_atoms, edge_index, edge_attr, 
                                       frag_index, x_frags, atom_to_frag_ids,
                                      node_feautures_bond_graph,edge_index_bonds_graph,edge_attr_bond_graph)
        x_atoms, x_frags = self.act(x_atoms), self.act(x_frags)
        
        x_atoms, x_frags = self.layer3(x_atoms, edge_index, edge_attr, 
                                       frag_index, x_frags, atom_to_frag_ids,
                                      node_feautures_bond_graph,edge_index_bonds_graph,edge_attr_bond_graph)
        x_atoms, x_frags = self.act(x_atoms), self.act(x_frags)
        
        x_atoms, x_frags = self.layer4(x_atoms, edge_index, edge_attr, 
                                       frag_index, x_frags, atom_to_frag_ids,
                                      node_feautures_bond_graph,edge_index_bonds_graph,edge_attr_bond_graph)
        x_atoms, x_frags = self.act(x_atoms), self.act(x_frags)
        
        
        return x_atoms, x_frags

    
class FragNetPreTrain(nn.Module):
    
    def __init__(self, n_classes):
        super(FragNetPreTrain, self).__init__()
        
        self.pretrain = FragNet(num_layer=6, drop_ratio=0.15)
        self.lin1 = nn.Linear(128, 128)
        self.out = nn.Linear(128, n_classes)
        self.dropout = nn.Dropout(p=0.15)
        self.activation = nn.ReLU()

        
    def forward(self, batch):
        
        x_atoms, x_frags = self.pretrain(batch)
        x = self.dropout(x_atoms)
        x = self.lin1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.out(x)
        
    
        return x
    
class FragNetFineTune(nn.Module):
    
    def __init__(self, n_classes=1, num_layer=6, drop_ratio=0.15):
        super(FragNetFineTune, self).__init__()
        

        self.pretrain = FragNet(num_layer=num_layer, drop_ratio=drop_ratio)
        self.lin1 = nn.Linear(128*2, 128*2)
        self.out = nn.Linear(128*2, n_classes)
        self.dropout = nn.Dropout(p=0.15)
        self.activation = nn.ReLU()
        
        
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
    


