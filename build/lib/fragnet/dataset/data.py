from .fragments import FragmentedMol
import torch
from torch_scatter import scatter_add
from rdkit import Chem
import numpy as np
from torch_geometric.data import Data
import random
from rdkit.Geometry import Point3D


def get_incr_atom_nodes(data_list):
    
    max_atoms=0
    zeros=[]
    if len(data_list) == 1:
        incr_atom_nodes = torch.zeros(data_list[0].edge_index.shape[1])
    else:
        for i in range(1, len(data_list)):

            max_atoms += data_list[i-1].x_atoms.size(0)
            zeros.append(torch.zeros(data_list[i].edge_index.shape[1]) + max_atoms)

        incr_atom_nodes = torch.cat(zeros)
        incr_atom_nodes = torch.cat([torch.zeros(data_list[0].edge_index.shape[1]), incr_atom_nodes], dim=0)
    return incr_atom_nodes


def get_incr_frag_nodes(data_list):
    
    max_atoms=0
    zeros=[]
    if len(data_list) == 1:
        incr_atom_nodes = torch.zeros(data_list[0].frag_index.shape[1])
    else:
        for i in range(1, len(data_list)):

            max_atoms += data_list[i-1].n_frags.item()
            zeros.append(torch.zeros(data_list[i].frag_index.shape[1]) + max_atoms)

        incr_atom_nodes = torch.cat(zeros)
        incr_atom_nodes = torch.cat([torch.zeros(data_list[0].frag_index.shape[1]), incr_atom_nodes], dim=0)
    return incr_atom_nodes


def get_incr_atom_id_frag_id_nodes(data_list):
    
    max_atoms=0
    zeros=[]
    if len(data_list) == 1:
        incr_atom_nodes = torch.zeros(data_list[0].atom_id_frag_id.shape[0])
    else:
        for i in range(1, len(data_list)):

            max_atoms += data_list[i-1].n_frags.item()
            zeros.append(torch.zeros(data_list[i].atom_id_frag_id.shape[0]) + max_atoms)

        incr_atom_nodes = torch.cat(zeros)
        incr_atom_nodes = torch.cat([torch.zeros(data_list[0].atom_id_frag_id.shape[0]), incr_atom_nodes], dim=0)
    return incr_atom_nodes

def get_incr_bond_nodes(data_list):
    
    max_bonds=0
    zeros=[]
    
    if len(data_list) == 1:
        incr_bond_nodes = torch.zeros(data_list[0].edge_index_bonds.shape[1])
        
    else:
        for i in range(1, len(data_list)):

            max_bonds += data_list[i-1].node_features_bonds.size(0)
            zeros.append(torch.zeros(data_list[i].edge_index_bonds.shape[1]) + max_bonds)

        incr_bond_nodes = torch.cat(zeros)
        incr_bond_nodes = torch.cat([torch.zeros(data_list[0].edge_index_bonds.shape[1]), incr_bond_nodes], dim=0)
    return incr_bond_nodes


def get_incr_fbond_nodes(data_list):

    max_bonds=0
    zeros=[]
    
    if len(data_list) == 1:
        incr_bond_nodes = torch.zeros(data_list[0].edge_index_fbondg.shape[1])
        
    else:
        for i in range(1, len(data_list)):

            max_bonds += data_list[i-1].node_feautures_fbondg.size(0)
            zeros.append(torch.zeros(data_list[i].edge_index_fbondg.shape[1]) + max_bonds)

        incr_bond_nodes = torch.cat(zeros)
        incr_bond_nodes = torch.cat([torch.zeros(data_list[0].edge_index_fbondg.shape[1]), incr_bond_nodes], dim=0)
    return incr_bond_nodes

    

def get_bond_pair_bond_graph(idx_bond_index):
    
    nnodes = len(idx_bond_index)
    res = [[],[]]
    for i in range(nnodes): 
        b1 = idx_bond_index[i]
        for j in range(nnodes):
            b2 = idx_bond_index[j]
            if len(list(set(b1).intersection(b2))) ==1:
                res[0] += [i]
                res[1] += [j]
                
    return res   

def get_bond_pair_fbond_graph(idx_bond_index):
    
    nnodes = len(idx_bond_index)
    res = [[],[]]

    if (nnodes==2):
        for i in range(nnodes): 
            b1 = idx_bond_index[i]
            for j in range(nnodes):
                b2 = idx_bond_index[j]
                if b1 != b2:
                    res[0] += [i]
                    res[1] += [j]
        
    else:
        for i in range(nnodes): 
            b1 = idx_bond_index[i]
            for j in range(nnodes):
                b2 = idx_bond_index[j]
                if len(list(set(b1).intersection(b2))) ==1 :
                    res[0] += [i]
                    res[1] += [j]
                
    return res   

def get_one_bond_frags(mol):    
    one_bond_frags=[]
    for frag in Chem.GetMolFrags(mol):
        if len(frag)==2:
            one_bond_frags.append(frag)
    return one_bond_frags

def add_one_bond_frag_nodes_to_index(edge_index_bond_graph, bond_index_to_id, one_bond_frags):
    
    nodes=[]
    for f in one_bond_frags:
        bond_id1 = bond_index_to_id[f]    
        bond_id2 = bond_index_to_id[  (f[1], f[0])  ]
        nodes.append((bond_id1, bond_id2))
        nodes.append((bond_id2, bond_id1))

        edge_index_bond_graph[0] += [bond_id1]
        edge_index_bond_graph[1] += [bond_id2]
        
        edge_index_bond_graph[0] += [bond_id2]
        edge_index_bond_graph[1] += [bond_id1]
        
    return edge_index_bond_graph, nodes

def get_edge_attr_bond_graph(edge_index, idx_bond_index, conf, one_bond_frag_ids=[]):
    edge_attr=[]
    for j in range(len(edge_index[0])):

        node1 = edge_index[0][j]
        node2 = edge_index[1][j]

        if (node1, node2) in one_bond_frag_ids:
            # this is for fragments with one bond. may have to experiment about this more.
            # how should we represent such fragments.
            edge_attr.append(1)
        else:
            # for the usual bonds
            atoms_in_n1 = idx_bond_index[ node1 ]
            atoms_in_n2 = idx_bond_index[ node2 ]

            common = list(set(atoms_in_n1).intersection(atoms_in_n2))[0]
            others = list(set(atoms_in_n1+atoms_in_n2)-set([common]))

            c = conf
            ang = Chem.rdMolTransforms.GetAngleRad(c, int(others[0]), int(common), int(others[1]) )
            cos_ang = np.cos(ang)
            edge_attr.append(cos_ang)
            
            
        
    return np.array(edge_attr).reshape(-1,1)


def fix_zero_pos(zero_pos, conf):    
    incr=0.0001
    for i in range(1, len(zero_pos)):
        conf.SetAtomPosition(int(zero_pos[i]),Point3D(0+incr, 0+incr, 0+incr))
        incr+=0.0001

# get bond angles;
# https://github.com/LARS-research/3D-PGT/blob/main/GEOM_dataset_preparation.py
# https://github.com/LARS-research/3D-PGT/blob/main/graphgps/head/pretrain_task.py
def get_bond_angle_dhangle(conf, x_atoms, edge_index):


    c = conf
    positions = c.GetPositions()
    positions = torch.tensor(positions, dtype=torch.float)

    
    bond_length_pair = positions[edge_index.T]
    bond_length_true = torch.sum((bond_length_pair[:, 0, :] - bond_length_pair[:, 1, :]) ** 2, axis=1)

    unit = (bond_length_pair[:,0,:] - bond_length_pair[:,1,:])
    unit_vector = unit / unit.norm(dim=-1).unsqueeze(1).repeat(1,3)
    direction_unit = torch.zeros(x_atoms.shape[0],3)
    for i in range(x_atoms.shape[0]):
        direction_unit[i] = unit_vector[edge_index[0] == i].sum()
    bond_angle_true = (direction_unit.norm(dim=-1)**2).unsqueeze(1)
    
    # dihedral_angle
    unit_neg = (bond_length_pair[:,1,:] - bond_length_pair[:,0,:])
    unit_neg_vector = unit_neg / unit_neg.norm(dim=-1).unsqueeze(1).repeat(1,3)
    
    dihedral_angle_true = torch.zeros(edge_index.shape[1])
    for i in range(edge_index.shape[1]):
        rej_pos = direction_unit[edge_index[0][i]] - torch.dot(direction_unit[edge_index[0][i]], unit_vector[i]) * unit_vector[i]
        rej_neg = direction_unit[edge_index[1][i]] - torch.dot(direction_unit[edge_index[0][i]], unit_neg_vector[i]) * unit_neg_vector[i]
        dihedral_angle_true[i] = torch.dot(rej_pos, rej_neg)


    return bond_length_true, bond_angle_true, dihedral_angle_true


def get_fragbond(frag_idx, cnx_attr):

    keys = frag_idx.T.numpy().tolist()

    # cnx_attr
    cnx_attr_list = cnx_attr.numpy().tolist()
    # cnx_attr_list
    frag_bond_cnx_attr = { tuple(keys[i]) : cnx_attr_list[i] for i in range(len(keys)) }
    # frag_bond_cnx_attr


    idx_fragbond_index = {}
    node_feautures_fragbond_graph = []
    for ie, i in enumerate(range(0,len(frag_idx[0]), 1)):
    
        id1 = frag_idx[0][i].item()
        id2 = frag_idx[1][i].item()
        idx_fragbond_index[ie] = [id1, id2]
    
        # bond = graph.mol.GetBondBetweenAtoms(id1, id2)
        frag_feature = frag_bond_cnx_attr[ (id1, id2) ]
        node_feautures_fragbond_graph.append(frag_feature)


    node_feautures_fragbond_graph = torch.tensor(node_feautures_fragbond_graph, dtype=torch.float)
    # fragbond_index_to_id = {tuple(v):i for i,v in idx_fragbond_index.items()}
    edge_index_fragbond_graph = get_bond_pair_fbond_graph(idx_fragbond_index)

    edge_attr_fragbond =[]
    for j in range(len(edge_index_fragbond_graph[0])):
        node1 = edge_index_fragbond_graph[0][j] # node ids for the frag graph
        node2 = edge_index_fragbond_graph[1][j]
    
    
        node_1_bond = idx_fragbond_index[ node1 ]
        node_2_bond = idx_fragbond_index[ node2 ]
    
        edge_feature = list( np.array(frag_bond_cnx_attr[tuple( node_1_bond  )]) + np.array(frag_bond_cnx_attr[tuple( node_2_bond )]) )
        edge_attr_fragbond.append( edge_feature )


    edge_index_fragbond_graph = torch.tensor(edge_index_fragbond_graph, dtype=torch.int32)
    edge_attr_fragbond = torch.tensor(edge_attr_fragbond, dtype=torch.float)



    return node_feautures_fragbond_graph, edge_index_fragbond_graph, edge_attr_fragbond


class CreateData:

    def __init__(self, data_type, create_bond_graph_data=True, add_dhangles=False,
                 create_fragbond_graph=True):

        self.feature_dtype = torch.float
        self.create_bond_graph_data = create_bond_graph_data
        self.add_dhangles = add_dhangles
        self.create_fragbond_graph = create_fragbond_graph

        if data_type in ['exp', 'exp1s']:
            from .features import FeaturesEXP
            self.feature_creator = FeaturesEXP()

        elif data_type in ['exp0', 'exp01s']:
            from .features0 import FeaturesEXP
            self.feature_creator = FeaturesEXP()

        if '1s' in data_type:
            self.get_frag_idx_cnx_attr = self.get_frag_idx_cnx_attr_1s
        else:
            self.get_frag_idx_cnx_attr = self.get_frag_idx_cnx_attr_2s


        
    def create_data_point(self, args):

        smiles = args[0]
        y = args[1]
        mol = args[2]
        conf = args[3]
        frag_type = args[4] 



        graph = FragmentedMol(mol, conf, frag_type)

        # cannot have two atoms at the same position. i assume that this occurs for [0,0,0] positions.
        # so, igonoring molecules where there are mutiple [0,0,0] coordinates.
        if np.count_nonzero([sum(i) == 0 for i in conf.GetPositions() ]) > 1:
            zero_pos = np.where([sum(i) == 0 for i in conf.GetPositions() ])[0]
            fix_zero_pos(zero_pos, conf)
            # return None
        

        x_atoms, edge_index, edge_attr = self.feature_creator.get_atom_and_bond_features_atom_graph_one_hot(graph.mol, self.feature_creator.use_bond_chirality)

        #### commmenting this section #### Ma4 28 2024    
        # no edges
        if len(edge_index[0])==0 or (not (len(x_atoms) == max(edge_index[0])+1 == max(edge_index[1])+1)):
            return None
        #### commmenting this section ####
        
        x_atoms = torch.tensor(np.array(x_atoms), dtype=self.feature_dtype)

        edge_index = torch.tensor(edge_index, dtype=torch.long) # edge index atom graph
        edge_attr = torch.tensor(edge_attr, dtype=self.feature_dtype)
        

        if self.create_bond_graph_data:
            
            idx_bond_index = {}
            node_feautures_bond_graph = []
            for ie, i in enumerate(range(0,len(edge_index[0]), 1)):

                id1 = edge_index[0][i].item()
                id2 = edge_index[1][i].item()
                idx_bond_index[ie] = [id1, id2]

                bond = graph.mol.GetBondBetweenAtoms(id1, id2)

                if bond:
                    node_feautures_bond_graph.append(self.feature_creator.bond_features_one_hot(bond, use_chirality=self.feature_creator.use_bond_chirality))
                # else:
                #     node_feautures_bond_graph.append(np.zeros((12)).tolist()) # TODO: TEST THIS
                    
                
            node_feautures_bond_graph = torch.tensor(node_feautures_bond_graph, dtype=torch.float)
            bond_index_to_id = {tuple(v):i for i,v in idx_bond_index.items()}

            edge_index_bond_graph = get_bond_pair_bond_graph(idx_bond_index)
            
            one_bond_frags = get_one_bond_frags(graph.mol)
            edge_index_bond_graph, one_bond_frag_ids = add_one_bond_frag_nodes_to_index(edge_index_bond_graph, bond_index_to_id, one_bond_frags)
            edge_attr_bond_graph = get_edge_attr_bond_graph(edge_index_bond_graph, idx_bond_index, conf, one_bond_frag_ids)

            edge_index_bond_graph = torch.tensor(edge_index_bond_graph, dtype=torch.int32)
            edge_attr_bond_graph = torch.tensor(edge_attr_bond_graph, dtype=torch.float)
            

        # for bond graph
        atom_id_frag_id = torch.tensor(list(graph.atom_to_frag_id.values()), dtype=torch.long)        
        x_frags = scatter_add(src=x_atoms, index=atom_id_frag_id, dim=0)
        

        frag_idx, cnx_attr = self.get_frag_idx_cnx_attr(graph)
        n_frags = len(graph.fragments)



        if self.create_fragbond_graph:
            node_feautures_fragbond_graph, edge_index_fragbond_graph, edge_attr_fragbond = get_fragbond(frag_idx, cnx_attr)
            
            

        if self.create_bond_graph_data:
            data = Data(x_atoms = x_atoms,
                        edge_index = edge_index,
                        edge_attr = edge_attr,
                        frag_index = frag_idx,
                        cnx_attr = cnx_attr,
                        x_frags = x_frags,
                        atom_id_frag_id = atom_id_frag_id,
                        n_frags = torch.tensor([n_frags], dtype=torch.long),
                        node_features_bonds = node_feautures_bond_graph,
                        edge_index_bonds = edge_index_bond_graph,
                        edge_attr_bonds = edge_attr_bond_graph,
                        y = torch.tensor(y, dtype=torch.float),
                        # y = y,
                        smiles=smiles,
            )

        else:
            data = Data(x_atoms = x_atoms,
                        edge_index = edge_index,
                        edge_attr = edge_attr,
                        frag_index = frag_idx,
                        cnx_attr = cnx_attr,
                        x_frags = x_frags,
                        atom_id_frag_id = atom_id_frag_id,
                        n_frags = torch.tensor([n_frags], dtype=torch.long),
                        y = torch.tensor(y, dtype=torch.float),
                        smiles=smiles,
                        )

        if self.create_fragbond_graph:

            data.node_feautures_fbondg   = node_feautures_fragbond_graph
            data.edge_index_fbondg   =  edge_index_fragbond_graph
            data.edge_attr_fbondg   = edge_attr_fragbond

        
        if self.add_dhangles:
            bond_length_pair, bond_angle_true, dihedral_angle_true = get_bond_angle_dhangle(conf, data.x_atoms, data.edge_index)
            data.bnd_lngth = bond_length_pair.reshape(-1,1)
            data.bnd_angl = bond_angle_true
            data.dh_angl = dihedral_angle_true.reshape(-1,1)
            

        return data
    


    def get_frag_idx_cnx_attr_2s(self, graph):
        frag_idx = [[],[]]
        cnx_attr = []
        for connection in graph.connections:
            frag_idx[0] += [connection.BeginFragIdx, connection.EndFragIdx]
            frag_idx[1] += [connection.EndFragIdx, connection.BeginFragIdx]
            cnx_attr.append(self.feature_creator.connection_features_one_hot(connection))    
            cnx_attr.append(self.feature_creator.connection_features_one_hot(connection))

        frag_idx = torch.tensor(frag_idx, dtype=torch.long)
        cnx_attr = torch.tensor(cnx_attr, dtype=self.feature_dtype)

        return frag_idx, cnx_attr
    

    def get_frag_idx_cnx_attr_1s(self, graph):
        frag_idx = [[],[]]
        cnx_attr = []

        n_frags = len(graph.fragments)

        if n_frags==1:
            for connection in graph.connections:
                frag_idx[0] += [connection.BeginFragIdx]
                frag_idx[1] += [connection.EndFragIdx]
                cnx_attr.append(self.feature_creator.connection_features_one_hot(connection))    
                # cnx_attr.append(connection_features_one_hot(connection))
        
        else:
            for connection in graph.connections:
                frag_idx[0] += [connection.BeginFragIdx, connection.EndFragIdx]
                frag_idx[1] += [connection.EndFragIdx, connection.BeginFragIdx]
                cnx_attr.append(self.feature_creator.connection_features_one_hot(connection))    
                cnx_attr.append(self.feature_creator.connection_features_one_hot(connection))


        frag_idx = torch.tensor(frag_idx, dtype=torch.long)
        cnx_attr = torch.tensor(cnx_attr, dtype=self.feature_dtype)

        return frag_idx, cnx_attr
    


class CreateDataDTA(CreateData):
    def __init__(self, data_type, create_bond_graph_data=True, add_dhangles=False,
                 create_fragbond_graph=True):
        super().__init__(data_type=data_type, create_bond_graph_data=create_bond_graph_data, 
                         add_dhangles=add_dhangles, create_fragbond_graph=create_fragbond_graph)


        seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
        seq_dict = {v:(i+1) for i,v in enumerate(seq_voc)}
        seq_dict_len = len(seq_dict)
        self.max_seq_len = 1000
        self.seq_dict = seq_dict


    def create_data_point(self, args):
        smiles = args[0]
        y = args[1]
        mol = args[2]
        conf = args[3] 
        protein = args[4]
        encoded_protein = self.encode_protein(protein)


        graph = FragmentedMol(mol, conf)

        # cannot have two atoms at the same position. i assume that this occurs for [0,0,0] positions.
        # so, igonoring molecules where there are mutiple [0,0,0] coordinates.
        if np.count_nonzero([sum(i) == 0 for i in conf.GetPositions() ]) > 1:
            zero_pos = np.where([sum(i) == 0 for i in conf.GetPositions() ])[0]
            fix_zero_pos(zero_pos, conf)
            # return None
        

        x_atoms, edge_index, edge_attr = self.feature_creator.get_atom_and_bond_features_atom_graph_one_hot(graph.mol, self.feature_creator.use_bond_chirality)

            
        # no edges
        if len(edge_index[0])==0 or (not (len(x_atoms) == max(edge_index[0])+1 == max(edge_index[1])+1)):
            return None
            
        x_atoms = torch.tensor(np.array(x_atoms), dtype=self.feature_dtype)

        edge_index = torch.tensor(edge_index, dtype=torch.long) # edge index atom graph
        edge_attr = torch.tensor(edge_attr, dtype=self.feature_dtype)
        
        # for bond graph
        if self.create_bond_graph_data:
            
            idx_bond_index = {}
            node_feautures_bond_graph = []
            for ie, i in enumerate(range(0,len(edge_index[0]), 1)):

                id1 = edge_index[0][i].item()
                id2 = edge_index[1][i].item()
                idx_bond_index[ie] = [id1, id2]

                bond = graph.mol.GetBondBetweenAtoms(id1, id2)

                if bond:
                    node_feautures_bond_graph.append(self.feature_creator.bond_features_one_hot(bond, use_chirality=self.feature_creator.use_bond_chirality))
                # else:
                #     node_feautures_bond_graph.append(np.zeros((12)).tolist()) # TODO: TEST THIS
                    
                
            node_feautures_bond_graph = torch.tensor(node_feautures_bond_graph, dtype=torch.float)
            bond_index_to_id = {tuple(v):i for i,v in idx_bond_index.items()}

            edge_index_bond_graph = get_bond_pair_bond_graph(idx_bond_index)
            
            one_bond_frags = get_one_bond_frags(graph.mol)
            edge_index_bond_graph, one_bond_frag_ids = add_one_bond_frag_nodes_to_index(edge_index_bond_graph, bond_index_to_id, one_bond_frags)
            edge_attr_bond_graph = get_edge_attr_bond_graph(edge_index_bond_graph, idx_bond_index, conf, one_bond_frag_ids)

            edge_index_bond_graph = torch.tensor(edge_index_bond_graph, dtype=torch.int32)
            edge_attr_bond_graph = torch.tensor(edge_attr_bond_graph, dtype=torch.float)
            

        # for bond graph
        atom_id_frag_id = torch.tensor(list(graph.atom_to_frag_id.values()), dtype=torch.long)        
        x_frags = scatter_add(src=x_atoms, index=atom_id_frag_id, dim=0)
        
        frag_idx, cnx_attr = self.get_frag_idx_cnx_attr(graph)
        n_frags = len(graph.fragments)

        if self.create_fragbond_graph:
            node_feautures_fragbond_graph, edge_index_fragbond_graph, edge_attr_fragbond = get_fragbond(frag_idx, cnx_attr)
            
            

        if self.create_bond_graph_data:
            data = Data(x_atoms = x_atoms,
                        edge_index = edge_index,
                        edge_attr = edge_attr,
                        frag_index = frag_idx,
                        cnx_attr = cnx_attr,
                        x_frags = x_frags,
                        atom_id_frag_id = atom_id_frag_id,
                        n_frags = torch.tensor([n_frags], dtype=torch.long),
                        node_features_bonds = node_feautures_bond_graph,
                        edge_index_bonds = edge_index_bond_graph,
                        edge_attr_bonds = edge_attr_bond_graph,
                        y = torch.tensor(y, dtype=torch.float),
                        smiles=smiles,
                        protein = encoded_protein
            )

        else:
            data = Data(x_atoms = x_atoms,
                        edge_index = edge_index,
                        edge_attr = edge_attr,
                        frag_index = frag_idx,
                        cnx_attr = cnx_attr,
                        x_frags = x_frags,
                        atom_id_frag_id = atom_id_frag_id,
                        n_frags = torch.tensor([n_frags], dtype=torch.long),
                        y = torch.tensor(y, dtype=torch.float),
                        smiles=smiles,
                        protein = encoded_protein
                        )

        if self.create_fragbond_graph:

            data.node_feautures_fbondg   = node_feautures_fragbond_graph
            data.edge_index_fbondg   =  edge_index_fragbond_graph
            data.edge_attr_fbondg   = edge_attr_fragbond

        
        if self.add_dhangles:
            bond_length_pair, bond_angle_true, dihedral_angle_true = get_bond_angle_dhangle(conf, data.x_atoms, data.edge_index)
            data.bnd_lngth = bond_length_pair.reshape(-1,1)
            data.bnd_angl = bond_angle_true
            data.dh_angl = dihedral_angle_true.reshape(-1,1)
            

        return data



    def seq_cat(self, prot):
        x = np.zeros(self.max_seq_len)
        for i, ch in enumerate(prot[:self.max_seq_len]):
            x[i] = self.seq_dict[ch]
        return x

    def encode_protein(self, protein):

        encoded = self.seq_cat(protein)
        encoded = torch.tensor(encoded, dtype=torch.float)
        
        return encoded


class CreateDataCDRP(CreateData):
    def __init__(self, data_type, create_bond_graph_data=True, add_dhangles=False,
                 create_fragbond_graph=True):
        super().__init__(data_type=data_type, create_bond_graph_data=create_bond_graph_data, 
                         add_dhangles=add_dhangles, create_fragbond_graph=create_fragbond_graph)




    def create_data_point(self, args):
  
        smiles = args[0]
        y = args[1]
        mol = args[2]
        conf = args[3] 
        gene_expr = args[4]


        gene_expr = torch.tensor(gene_expr, dtype=torch.float)
  

        graph = FragmentedMol(mol, conf)

        # cannot have two atoms at the same position. i assume that this occurs for [0,0,0] positions.
        # so, igonoring molecules where there are mutiple [0,0,0] coordinates.
        if np.count_nonzero([sum(i) == 0 for i in conf.GetPositions() ]) > 1:
            zero_pos = np.where([sum(i) == 0 for i in conf.GetPositions() ])[0]
            fix_zero_pos(zero_pos, conf)

        x_atoms, edge_index, edge_attr = self.feature_creator.get_atom_and_bond_features_atom_graph_one_hot(graph.mol, self.feature_creator.use_bond_chirality)

            
        # no edges
        if len(edge_index[0])==0 or (not (len(x_atoms) == max(edge_index[0])+1 == max(edge_index[1])+1)):
            return None

            
        x_atoms = torch.tensor(np.array(x_atoms), dtype=self.feature_dtype)

        edge_index = torch.tensor(edge_index, dtype=torch.long) # edge index atom graph
        edge_attr = torch.tensor(edge_attr, dtype=self.feature_dtype)
        
        # for bond graph
        # create_bond_graph_data=True
        if self.create_bond_graph_data:
            
            idx_bond_index = {}
            node_feautures_bond_graph = []
            for ie, i in enumerate(range(0,len(edge_index[0]), 1)):

                id1 = edge_index[0][i].item()
                id2 = edge_index[1][i].item()
                idx_bond_index[ie] = [id1, id2]

                bond = graph.mol.GetBondBetweenAtoms(id1, id2)

                if bond:
                    node_feautures_bond_graph.append(self.feature_creator.bond_features_one_hot(bond, use_chirality=self.feature_creator.use_bond_chirality))
                # else:
                #     node_feautures_bond_graph.append(np.zeros((12)).tolist()) # TODO: TEST THIS
                    
                
            node_feautures_bond_graph = torch.tensor(node_feautures_bond_graph, dtype=torch.float)
            bond_index_to_id = {tuple(v):i for i,v in idx_bond_index.items()}

            edge_index_bond_graph = get_bond_pair_bond_graph(idx_bond_index)
            
            one_bond_frags = get_one_bond_frags(graph.mol)
            edge_index_bond_graph, one_bond_frag_ids = add_one_bond_frag_nodes_to_index(edge_index_bond_graph, bond_index_to_id, one_bond_frags)
            edge_attr_bond_graph = get_edge_attr_bond_graph(edge_index_bond_graph, idx_bond_index, conf, one_bond_frag_ids)

            edge_index_bond_graph = torch.tensor(edge_index_bond_graph, dtype=torch.int32)
            edge_attr_bond_graph = torch.tensor(edge_attr_bond_graph, dtype=torch.float)
            

        # for bond graph
        atom_id_frag_id = torch.tensor(list(graph.atom_to_frag_id.values()), dtype=torch.long)        
        x_frags = scatter_add(src=x_atoms, index=atom_id_frag_id, dim=0)
        

        frag_idx, cnx_attr = self.get_frag_idx_cnx_attr(graph)
        n_frags = len(graph.fragments)

        if self.create_fragbond_graph:
            node_feautures_fragbond_graph, edge_index_fragbond_graph, edge_attr_fragbond = get_fragbond(frag_idx, cnx_attr)
            
            

        if self.create_bond_graph_data:
            data = Data(x_atoms = x_atoms,
                        edge_index = edge_index,
                        edge_attr = edge_attr,
                        frag_index = frag_idx,
                        cnx_attr = cnx_attr,
                        x_frags = x_frags,
                        atom_id_frag_id = atom_id_frag_id,
                        n_frags = torch.tensor([n_frags], dtype=torch.long),
                        node_features_bonds = node_feautures_bond_graph,
                        edge_index_bonds = edge_index_bond_graph,
                        edge_attr_bonds = edge_attr_bond_graph,
                        y = torch.tensor(y, dtype=torch.float),
                        # y = y,
                        smiles=smiles,
                        gene_expr = gene_expr
            )

        else:
            data = Data(x_atoms = x_atoms,
                        edge_index = edge_index,
                        edge_attr = edge_attr,
                        frag_index = frag_idx,
                        cnx_attr = cnx_attr,
                        x_frags = x_frags,
                        atom_id_frag_id = atom_id_frag_id,
                        n_frags = torch.tensor([n_frags], dtype=torch.long),
                        y = torch.tensor(y, dtype=torch.float),
                        smiles=smiles,
                        gene_expr = gene_expr
                        )

        if self.create_fragbond_graph:

            data.node_feautures_fbondg   = node_feautures_fragbond_graph
            data.edge_index_fbondg   =  edge_index_fragbond_graph
            data.edge_attr_fbondg   = edge_attr_fragbond

        
        if self.add_dhangles:
            bond_length_pair, bond_angle_true, dihedral_angle_true = get_bond_angle_dhangle(conf, data.x_atoms, data.edge_index)
            data.bnd_lngth = bond_length_pair.reshape(-1,1)
            data.bnd_angl = bond_angle_true
            data.dh_angl = dihedral_angle_true.reshape(-1,1)
            

        return data



def collate_fn(data_list):
    # atom features
    x_atoms_batch = torch.cat([i.x_atoms for i in data_list], dim=0)
    # bond indices
    edge_index = torch.cat([i.edge_index for i in data_list], dim=1)
    incr_atom_nodes = get_incr_atom_nodes(data_list)
    edge_index = edge_index + incr_atom_nodes

    edge_attr = torch.cat([i.edge_attr for i in data_list], dim=0)
    cnx_attr = torch.cat([i.cnx_attr for i in data_list], dim=0)
    
    
    frag_index = torch.cat([i.frag_index for i in data_list], dim=1)
    incr_frag_nodes = get_incr_frag_nodes(data_list)
    frag_index = frag_index + incr_frag_nodes

    x_frags = torch.cat([i.x_frags for i in data_list], dim=0)
    
    batch = torch.cat([torch.zeros(data_list[i].x_atoms.shape[0]) + i for i in range(len(data_list)) ])    
    frag_batch = torch.cat([torch.zeros(data_list[i].n_frags.item() ) + i for i in range(len(data_list)) ])
    
    
    atom_to_frag_ids = torch.cat([i.atom_id_frag_id for i in data_list], dim=0)
    incr_atomfrag_ids = get_incr_atom_id_frag_id_nodes(data_list)
    atom_to_frag_ids = atom_to_frag_ids + incr_atomfrag_ids
    
    
    
    # for bond graph
    node_features_bonds = torch.cat([i.node_features_bonds for i in data_list], dim=0)
    
    edge_index_bonds_graph = torch.cat([i.edge_index_bonds for i in data_list], dim=1)
    
    incr_bonds_nodes=get_incr_bond_nodes(data_list)
    edge_index_bonds_graph = edge_index_bonds_graph + incr_bonds_nodes
    
    edge_attr_bonds = torch.cat([i.edge_attr_bonds for i in data_list], dim=0)
    # for bond graph
    

    # for frag graph
    node_features_fragbonds = torch.cat([i.node_feautures_fbondg for i in data_list], dim=0)
    edge_index_fragbonds = torch.cat([i.edge_index_fbondg for i in data_list], dim=1)  
    
    incr_bonds_nodes = get_incr_fbond_nodes(data_list)   
    incr_bonds_nodes = incr_bonds_nodes.to(torch.long) 
    edge_index_fragbonds = edge_index_fragbonds + incr_bonds_nodes

    edge_attr_fragbonds = torch.cat([i.edge_attr_fbondg for i in data_list], dim=0)
    # for frag graphs

    

    
    y = torch.cat([i.y for i in data_list], dim=0)
    
    return {'x_atoms': x_atoms_batch,
           'edge_index': edge_index.type(torch.long),
            'frag_index': frag_index.type(torch.long),
            'x_frags': x_frags,
            'edge_attr': edge_attr,
            'cnx_attr' : cnx_attr,
            'batch':batch.type(torch.long),
            'frag_batch':frag_batch.type(torch.long),
            'atom_to_frag_ids': atom_to_frag_ids.type(torch.long),
            'node_features_bonds':node_features_bonds,
            'edge_index_bonds_graph':edge_index_bonds_graph.type(torch.long),
            'edge_attr_bonds':edge_attr_bonds,
            
            'node_features_fbonds':node_features_fragbonds,
            'edge_index_fbonds':edge_index_fragbonds,
            'edge_attr_fbonds': edge_attr_fragbonds,
            
            'y': y.type(torch.float)
            }


def collate_fn_pt(data_list):
    # atom features
    x_atoms_batch = torch.cat([i.x_atoms for i in data_list], dim=0)
    # bond indices
    edge_index = torch.cat([i.edge_index for i in data_list], dim=1)
    incr_atom_nodes = get_incr_atom_nodes(data_list)
    edge_index = edge_index + incr_atom_nodes

    edge_attr = torch.cat([i.edge_attr for i in data_list], dim=0)
    cnx_attr = torch.cat([i.cnx_attr for i in data_list], dim=0)
    
    
    frag_index = torch.cat([i.frag_index for i in data_list], dim=1)
    incr_frag_nodes = get_incr_frag_nodes(data_list)
    frag_index = frag_index + incr_frag_nodes

    x_frags = torch.cat([i.x_frags for i in data_list], dim=0)
    
    batch = torch.cat([torch.zeros(data_list[i].x_atoms.shape[0]) + i for i in range(len(data_list)) ])    
    frag_batch = torch.cat([torch.zeros(data_list[i].n_frags.item() ) + i for i in range(len(data_list)) ])
    
    
    atom_to_frag_ids = torch.cat([i.atom_id_frag_id for i in data_list], dim=0)
    incr_atomfrag_ids = get_incr_atom_id_frag_id_nodes(data_list)
    atom_to_frag_ids = atom_to_frag_ids + incr_atomfrag_ids
    
    
    
    # for bond graph
    node_features_bonds = torch.cat([i.node_features_bonds for i in data_list], dim=0)
    
    edge_index_bonds_graph = torch.cat([i.edge_index_bonds for i in data_list], dim=1)
    
    incr_bonds_nodes=get_incr_bond_nodes(data_list)
    edge_index_bonds_graph = edge_index_bonds_graph + incr_bonds_nodes
    
    edge_attr_bonds = torch.cat([i.edge_attr_bonds for i in data_list], dim=0)
    # for bond graph

    # for frag graph
    node_features_fragbonds = torch.cat([i.node_feautures_fbondg for i in data_list], dim=0)
    edge_index_fragbonds = torch.cat([i.edge_index_fbondg for i in data_list], dim=1)  
    
    incr_bonds_nodes = get_incr_fbond_nodes(data_list)   
    incr_bonds_nodes = incr_bonds_nodes.to(torch.long) 
    edge_index_fragbonds = edge_index_fragbonds + incr_bonds_nodes

    edge_attr_fragbonds = torch.cat([i.edge_attr_fbondg for i in data_list], dim=0)
    # for frag graphs

    # for pretrain tasks
    bnd_lngth_batch = torch.cat([i.bnd_lngth for i in data_list], dim=0)
    
    bnd_angl_batch = torch.cat([i.bnd_angl for i in data_list], dim=0)
    dh_angl_batch = torch.cat([i.dh_angl for i in data_list], dim=0)
    # for pretrain tasks
        

    y = torch.cat([i.y for i in data_list], dim=0)
    
    return {'x_atoms': x_atoms_batch,
           'edge_index': edge_index.type(torch.long),
            'frag_index': frag_index.type(torch.long),
            'x_frags': x_frags,
            'edge_attr': edge_attr,
            'cnx_attr' : cnx_attr,
            'batch':batch.type(torch.long),
            'frag_batch':frag_batch.type(torch.long),
            'atom_to_frag_ids': atom_to_frag_ids.type(torch.long),
            'node_features_bonds':node_features_bonds,
            'edge_index_bonds_graph':edge_index_bonds_graph.type(torch.long),
            'edge_attr_bonds':edge_attr_bonds,

            'node_features_fbonds':node_features_fragbonds,
            'edge_index_fbonds':edge_index_fragbonds,
            'edge_attr_fbonds': edge_attr_fragbonds,

            'bnd_lngth': bnd_lngth_batch,
            'bnd_angl': bnd_angl_batch,
            'dh_angl': dh_angl_batch,
            'y': y.type(torch.float)
            }






def collate_fn_dta(data_list):
    # atom features
    x_atoms_batch = torch.cat([i.x_atoms for i in data_list], dim=0)
    # bond indices
    edge_index = torch.cat([i.edge_index for i in data_list], dim=1)
    incr_atom_nodes = get_incr_atom_nodes(data_list)
    edge_index = edge_index + incr_atom_nodes

    edge_attr = torch.cat([i.edge_attr for i in data_list], dim=0)
    cnx_attr = torch.cat([i.cnx_attr for i in data_list], dim=0)
    
    
    frag_index = torch.cat([i.frag_index for i in data_list], dim=1)
    incr_frag_nodes = get_incr_frag_nodes(data_list)
    frag_index = frag_index + incr_frag_nodes

    x_frags = torch.cat([i.x_frags for i in data_list], dim=0)
    
    batch = torch.cat([torch.zeros(data_list[i].x_atoms.shape[0]) + i for i in range(len(data_list)) ])    
    frag_batch = torch.cat([torch.zeros(data_list[i].n_frags.item() ) + i for i in range(len(data_list)) ])
    
    
    atom_to_frag_ids = torch.cat([i.atom_id_frag_id for i in data_list], dim=0)
    incr_atomfrag_ids = get_incr_atom_id_frag_id_nodes(data_list)
    atom_to_frag_ids = atom_to_frag_ids + incr_atomfrag_ids
    
    
    
    # for bond graph
    node_features_bonds = torch.cat([i.node_features_bonds for i in data_list], dim=0)
    
    edge_index_bonds_graph = torch.cat([i.edge_index_bonds for i in data_list], dim=1)
    
    incr_bonds_nodes=get_incr_bond_nodes(data_list)
    edge_index_bonds_graph = edge_index_bonds_graph + incr_bonds_nodes
    
    edge_attr_bonds = torch.cat([i.edge_attr_bonds for i in data_list], dim=0)

    # for bond graph
    

    # for frag graph
    node_features_fragbonds = torch.cat([i.node_feautures_fbondg for i in data_list], dim=0)
    edge_index_fragbonds = torch.cat([i.edge_index_fbondg for i in data_list], dim=1)  
    
    incr_bonds_nodes = get_incr_fbond_nodes(data_list)   
    incr_bonds_nodes = incr_bonds_nodes.to(torch.long) 
    edge_index_fragbonds = edge_index_fragbonds + incr_bonds_nodes

    edge_attr_fragbonds = torch.cat([i.edge_attr_fbondg for i in data_list], dim=0)
    # for frag graphs

    

    
    y = torch.cat([i.y for i in data_list], dim=0)
    protein = torch.cat([ i.protein.view(1,-1) for i in data_list], dim=0)
    
    return {'x_atoms': x_atoms_batch,
           'edge_index': edge_index.type(torch.long),
            'frag_index': frag_index.type(torch.long),
            'x_frags': x_frags,
            'edge_attr': edge_attr,
            'cnx_attr' : cnx_attr,
            'batch':batch.type(torch.long),
            'frag_batch':frag_batch.type(torch.long),
            'atom_to_frag_ids': atom_to_frag_ids.type(torch.long),
            'node_features_bonds':node_features_bonds,
            'edge_index_bonds_graph':edge_index_bonds_graph.type(torch.long),
            'edge_attr_bonds':edge_attr_bonds,
            
            'node_features_fbonds':node_features_fragbonds,
            'edge_index_fbonds':edge_index_fragbonds,
            'edge_attr_fbonds': edge_attr_fragbonds,
            
            'y': y.type(torch.float),
            'protein': protein.type(torch.long)
            }


def collate_fn_cdrp(data_list):
    # atom features
    x_atoms_batch = torch.cat([i.x_atoms for i in data_list], dim=0)
    # bond indices
    edge_index = torch.cat([i.edge_index for i in data_list], dim=1)
    incr_atom_nodes = get_incr_atom_nodes(data_list)
    edge_index = edge_index + incr_atom_nodes

    edge_attr = torch.cat([i.edge_attr for i in data_list], dim=0)
    cnx_attr = torch.cat([i.cnx_attr for i in data_list], dim=0)
    
    
    frag_index = torch.cat([i.frag_index for i in data_list], dim=1)
    incr_frag_nodes = get_incr_frag_nodes(data_list)
    frag_index = frag_index + incr_frag_nodes

    x_frags = torch.cat([i.x_frags for i in data_list], dim=0)
    
    batch = torch.cat([torch.zeros(data_list[i].x_atoms.shape[0]) + i for i in range(len(data_list)) ])    
    frag_batch = torch.cat([torch.zeros(data_list[i].n_frags.item() ) + i for i in range(len(data_list)) ])
    
    
    atom_to_frag_ids = torch.cat([i.atom_id_frag_id for i in data_list], dim=0)
    incr_atomfrag_ids = get_incr_atom_id_frag_id_nodes(data_list)
    atom_to_frag_ids = atom_to_frag_ids + incr_atomfrag_ids
    
    
    
    # for bond graph
    node_features_bonds = torch.cat([i.node_features_bonds for i in data_list], dim=0)
    
    edge_index_bonds_graph = torch.cat([i.edge_index_bonds for i in data_list], dim=1)
    
    incr_bonds_nodes=get_incr_bond_nodes(data_list)
    edge_index_bonds_graph = edge_index_bonds_graph + incr_bonds_nodes
    
    edge_attr_bonds = torch.cat([i.edge_attr_bonds for i in data_list], dim=0)

    # for bond graph
    

    # for frag graph
    node_features_fragbonds = torch.cat([i.node_feautures_fbondg for i in data_list], dim=0)
    edge_index_fragbonds = torch.cat([i.edge_index_fbondg for i in data_list], dim=1)  
    
    incr_bonds_nodes = get_incr_fbond_nodes(data_list)   
    incr_bonds_nodes = incr_bonds_nodes.to(torch.long) 
    edge_index_fragbonds = edge_index_fragbonds + incr_bonds_nodes

    edge_attr_fragbonds = torch.cat([i.edge_attr_fbondg for i in data_list], dim=0)
    # for frag graphs

    

    
    y = torch.cat([i.y for i in data_list], dim=0)
    gene_expr = torch.cat([ i.gene_expr.view(1,-1) for i in data_list], dim=0)
    
    return {'x_atoms': x_atoms_batch,
           'edge_index': edge_index.type(torch.long),
            'frag_index': frag_index.type(torch.long),
            'x_frags': x_frags,
            'edge_attr': edge_attr,
            'cnx_attr' : cnx_attr,
            'batch':batch.type(torch.long),
            'frag_batch':frag_batch.type(torch.long),
            'atom_to_frag_ids': atom_to_frag_ids.type(torch.long),
            'node_features_bonds':node_features_bonds,
            'edge_index_bonds_graph':edge_index_bonds_graph.type(torch.long),
            'edge_attr_bonds':edge_attr_bonds,
            
            'node_features_fbonds':node_features_fragbonds,
            'edge_index_fbonds':edge_index_fragbonds,
            'edge_attr_fbonds': edge_attr_fragbonds,
            
            'y': y.type(torch.float),
            'gene_expr': gene_expr.type(torch.long)
            }


def mask_atom_features(batch):
    
    nmasks = round(batch['x_atoms'].shape[0]*.3)
    mask_ids = random.sample(range(batch['x_atoms'].shape[0]), nmasks )
    batch['x_atoms'][mask_ids, :] = torch.ones_like(batch['x_atoms'][mask_ids, :])*-1








