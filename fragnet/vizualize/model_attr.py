import torch
import torch.nn as nn
from fragnet.model.gat.gat2 import FragNetLayerA
from fragnet.model.gat.gat2 import FragNetLayerA, FTHead1, FTHead2, FTHead3, FTHead4
from torch_scatter import scatter_add
from torch.utils.data import DataLoader
from fragnet.dataset.dataset import FinetuneData
import pandas as pd
from fragnet.dataset.utils import extract_data
import matplotlib.cm as cm
import matplotlib



class FragNet(nn.Module):

    def __init__(self, num_layer, drop_ratio = 0.2, emb_dim=128, 
                 atom_features=167, frag_features=167, edge_features=16, fedge_in=6, fbond_edge_in=6, num_heads=4,
                apply_mask=False):
        super().__init__()
        self.num_layer = num_layer
        self.dropout = nn.Dropout(p=drop_ratio)
        self.act = nn.ReLU()

        self.apply_mask = apply_mask
# atom_features=45/66/75

        ###List of MLPs
#         self.gnns = torch.nn.ModuleList()
#         self.gnns.append(FragNetLayer(atom_in=66, atom_out=emb_dim, frag_in=66, frag_out=emb_dim,
#                  edge_in=12, edge_out=emb_dim))
#         for layer in range(num_layer):
#             self.gnns.append(FragNetLayer(atom_in=128, atom_out=emb_dim, frag_in=128, 
#                                                        frag_out=emb_dim, edge_in=12, edge_out=emb_dim)
#                                          )

        self.layers = torch.nn.ModuleList()
        self.layers.append(FragNetLayerA(atom_in=atom_features, atom_out=emb_dim, frag_in=frag_features, 
                                   frag_out=emb_dim, edge_in=edge_features, fedge_in=fedge_in, fbond_edge_in=fbond_edge_in, edge_out=emb_dim, num_heads=num_heads))
        
        # FragNetLayerA(atom_in=128, atom_out=128, frag_in=128, frag_out=128,
        #          edge_in=128, edge_out=128, fedge_in=128, num_heads=4)
        for i in range(num_layer-1):
            self.layers.append(FragNetLayerA(atom_in=emb_dim, atom_out=emb_dim, frag_in=emb_dim, 
                                   frag_out=emb_dim, edge_in=emb_dim, edge_out=emb_dim, fedge_in=emb_dim,
                                             fbond_edge_in=fbond_edge_in,
                                    num_heads=num_heads))
        # self.layer1 = FragNetLayerA(atom_in=45, atom_out=emb_dim, frag_in=45, 
        #                            frag_out=emb_dim, edge_in=edge_features, edge_out=emb_dim, num_heads=num_heads)
        # self.layer2 = FragNetLayerA(atom_in=emb_dim, atom_out=emb_dim, frag_in=emb_dim, 
        #                            frag_out=emb_dim, edge_in=emb_dim, edge_out=emb_dim, num_heads=num_heads)
        # self.layer3 = FragNetLayerA(atom_in=emb_dim, atom_out=emb_dim, frag_in=emb_dim, 
        #                            frag_out=emb_dim, edge_in=emb_dim, edge_out=emb_dim, num_heads=num_heads)
        # self.layer4 = FragNetLayerA(atom_in=emb_dim, atom_out=emb_dim, frag_in=emb_dim, 
        #                            frag_out=emb_dim, edge_in=emb_dim, edge_out=emb_dim, num_heads=num_heads)

        ###List of batchnorms
        # self.batch_norms_xa = torch.nn.ModuleList()
        # self.batch_norms_xf = torch.nn.ModuleList()
        # self.batch_norms_xe = torch.nn.ModuleList()
        # for layer in range(num_layer):
        #     self.batch_norms_xa.append(BatchNorm(emb_dim, track_running_stats=False))
        #     self.batch_norms_xf.append(BatchNorm(emb_dim, track_running_stats=False))
        #     self.batch_norms_xe.append(BatchNorm(emb_dim, track_running_stats=False))

    #def forward(self, x, edge_index, edge_attr):
    def forward(self, batch):
        
        
        x_atoms = batch['x_atoms']
        edge_index = batch['edge_index']
        frag_index = batch['frag_index']
        
        x_frags = batch['x_frags']
        edge_attr = batch['edge_attr']
        # atom_batch = batch['batch']
        # frag_batch = batch['frag_batch']
        atom_to_frag_ids = batch['atom_to_frag_ids']
        
        node_feautures_bond_graph=batch['node_features_bonds']
        edge_index_bonds_graph=batch['edge_index_bonds_graph']
        edge_attr_bond_graph=batch['edge_attr_bonds']


        node_feautures_fbondg = batch['node_features_fbonds']
        edge_index_fbondg = batch['edge_index_fbonds']
        edge_attr_fbondg = batch['edge_attr_fbonds']


        atom_mask = batch['atom_mask']
        # MASKING
        # if self.apply_mask:
        #     x_atoms[atom_mask==1] = 0.0 # maksing 

        
        x_atoms = self.dropout(x_atoms)
        x_frags = self.dropout(x_frags)
        
        # before passing through each layer, the features of masked atoms or bonds should be set equal to zero.

    
        x_atoms, x_frags, edge_features, fedge_features = self.layers[0](x_atoms, edge_index, edge_attr, 
                               frag_index, x_frags, atom_to_frag_ids,
                               node_feautures_bond_graph,edge_index_bonds_graph,edge_attr_bond_graph,
                               node_feautures_fbondg,edge_index_fbondg,edge_attr_fbondg
                               )
        # x_atoms, x_frags, edge_features = self.batch_norms_xa[0](x_atoms), self.batch_norms_xf[0](x_frags), self.batch_norms_xe[0](edge_features)
        
        
        x_atoms, x_frags = self.act(self.dropout(x_atoms)), self.act(self.dropout(x_frags))
        edge_features = self.act(self.dropout(edge_features))
        fedge_features = self.act(self.dropout(fedge_features))

        # MASKING
        # if self.apply_mask:
        #     x_atoms[atom_mask==1] = 0.0 # maksing 

        # i=1
        for layer in self.layers[1:]:
            x_atoms, x_frags, edge_features, fedge_features = layer(x_atoms, edge_index, edge_features, 
                                       frag_index, x_frags, atom_to_frag_ids,
                                      edge_features, edge_index_bonds_graph, edge_attr_bond_graph,
                                      fedge_features, edge_index_fbondg, edge_attr_fbondg
                                      )
            
            # x_atoms, x_frags, edge_features = self.batch_norms_xa[i](x_atoms), self.batch_norms_xf[i](x_frags), self.batch_norms_xe[i](edge_features)
            # i+=1
            
            x_atoms, x_frags = self.act(self.dropout(x_atoms)), self.act(self.dropout(x_frags))
            edge_features = self.act(self.dropout(edge_features))
            fedge_features = self.act(self.dropout(fedge_features))

            # MASKING        
            # if self.apply_mask:
            #     x_atoms[atom_mask==1] = 0.0 # maksing 
            

        
        return x_atoms, x_frags, edge_features, fedge_features

    
    
class FragNetFineTune(nn.Module):
    
    def __init__(self, n_classes=1, atom_features=167, frag_features=167, edge_features=17, 
                 num_layer=4, num_heads=4, drop_ratio=0.15,
                h1=256, h2=256, h3=256, h4=256, act='celu',emb_dim=128, fthead='FTHead3',
                apply_mask=False):
        super().__init__()
        

        self.pretrain = FragNet(num_layer=num_layer, drop_ratio=drop_ratio, 
                                num_heads=num_heads, emb_dim=emb_dim,
                                atom_features=atom_features, frag_features=frag_features, edge_features=edge_features,
                               apply_mask=apply_mask)
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
                                  input_dim=emb_dim,
                             h1=h1, h2=h2, h3=h3, h4=h4,
                             drop_ratio=drop_ratio, act=act)
            
        elif fthead == 'FTHead4':
            print('using FTHead4' )
            self.fthead = FTHead4(n_classes=n_classes,
                             h1=h1, drop_ratio=drop_ratio, act=act)
        

        self.apply_mask = apply_mask
        
    def forward(self, batch):
        
        x_atoms, x_frags, x_edge, x_fedge = self.pretrain(batch)


        if self.apply_mask:
            mask = batch['atom_mask']
            x_atoms[mask==1] = 0.0
        
        # do_nothing(x_edge)
        # do_nothing(x_fedge)

        x_frags_pooled = scatter_add(src=x_frags, index=batch['frag_batch'], dim=0)
        x_atoms_pooled = scatter_add(src=x_atoms, index=batch['batch'], dim=0)
        
        cat = torch.cat((x_atoms_pooled, x_frags_pooled), 1)
        # x = self.dropout(cat)
        # x = self.lin1(x)
        # x = self.activation(x)
        # x = self.dropout(x)
        # x = self.out(x)
        x = self.fthead(cat)
        
    
        return x



import torch
from fragnet.dataset.data import get_incr_atom_id_frag_id_nodes, get_incr_atom_nodes, get_incr_bond_nodes, get_incr_frag_nodes, get_incr_fbond_nodes

def collate_fn(data_list):
    # atom features
    x_atoms_batch = torch.cat([i.x_atoms for i in data_list], dim=0)
    atoms_mask_batch = torch.cat([i.atom_mask for i in data_list], dim=0)
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
            
            'y': y.type(torch.float),
            'atom_mask': atoms_mask_batch.type(torch.int)
            
            }


def get_predictions(dataset, model_config, chkpt_path):
    
    opt = OmegaConf.load(model_config)
    OmegaConf.resolve(opt)
    # opt.update(vars(args))
    args=opt

    # ds1 = copy.deepcopy(dataset)
    # ds2 = copy.deepcopy(dataset)
    
    # if model_type in ['property', 'cdrp']:
    model_no_mask = FragNetFineTune(n_classes=args.finetune.model.n_classes, 
                                atom_features=args.atom_features, 
                                frag_features=args.frag_features, 
                                edge_features=args.edge_features,
                                num_layer=args.finetune.model.num_layer, 
                                drop_ratio=args.finetune.model.drop_ratio,
                                    num_heads=args.finetune.model.num_heads, 
                                    emb_dim=args.finetune.model.emb_dim,
                                    h1=args.finetune.model.h1,
                                    h2=args.finetune.model.h2,
                                    h3=args.finetune.model.h3,
                                    h4=args.finetune.model.h4,
                                    act=args.finetune.model.act,
                                    fthead=args.finetune.model.fthead)
    
    # model_no_mask.load_state_dict(torch.load('../../exps/ft/pnnl_set2/fragnet_hpdl_exp1s_h4pt4_10/ft_100.pt', 
    #                                  map_location=torch.device('cpu') ))
    model_no_mask.load_state_dict(torch.load(chkpt_path, 
                                     map_location=torch.device('cpu') ))
    
    test_loader = DataLoader(dataset, collate_fn=collate_fn, batch_size=len(dataset), shuffle=False, drop_last=False)
    # test_loader = DataLoader(ds1, collate_fn=collate_fn, batch_size=len(dataset), shuffle=False, drop_last=False)
    batch = next(iter(test_loader))
    
    
    with torch.no_grad():
        model_no_mask.eval();
        pred_no_mask = model_no_mask(batch)
    
    
    model_mask = FragNetFineTune(n_classes=args.finetune.model.n_classes, 
                                atom_features=args.atom_features, 
                                frag_features=args.frag_features, 
                                edge_features=args.edge_features,
                                num_layer=args.finetune.model.num_layer, 
                                drop_ratio=args.finetune.model.drop_ratio,
                                    num_heads=args.finetune.model.num_heads, 
                                    emb_dim=args.finetune.model.emb_dim,
                                    h1=args.finetune.model.h1,
                                    h2=args.finetune.model.h2,
                                    h3=args.finetune.model.h3,
                                    h4=args.finetune.model.h4,
                                    act=args.finetune.model.act,
                                    fthead=args.finetune.model.fthead,
                           apply_mask=True)
    
    # model_mask.load_state_dict(torch.load('../../exps/ft/pnnl_set2/fragnet_hpdl_exp1s_h4pt4_10/ft_100.pt', 
    #                                  map_location=torch.device('cpu') ))
    model_mask.load_state_dict(torch.load(chkpt_path, 
                                     map_location=torch.device('cpu') ))
    
    
    test_loader = DataLoader(dataset, collate_fn=collate_fn, batch_size=len(dataset), shuffle=False, drop_last=False)
    # test_loader = DataLoader(ds2, collate_fn=collate_fn, batch_size=len(dataset), shuffle=False, drop_last=False)
    batch = next(iter(test_loader))
    
    
    
    with torch.no_grad():
        model_mask.eval();
        pred_mask = model_mask(batch)
    

    return pred_no_mask, pred_mask, batch['y']


import torch
import copy
from omegaconf import OmegaConf
from fragnet.dataset.fragments import get_3Dcoords
from fragnet.dataset.fragments import FragmentedMol
from fragnet.vizualize.viz import get_frags, get_atoms_in_frags

def add_atom_numbers(mol):
        
    for atom in mol.GetAtoms():
        # For each atom, set the property "atomNote" to a index+1 of the atom
        atom.SetProp("atomNote", str(atom.GetIdx()))

def get_mol(smiles):
    mol = get_3Dcoords(smiles)
    conf = mol.GetConformer(id=0)
    frag = FragmentedMol(mol, conf)
    mol = frag.mol
    mol.RemoveAllConformers()
    # self.mol = mol
    add_atom_numbers(mol)
    return mol



def create_data(data):
    
    copies=[]
    for i, item in enumerate(data):
        smiles = item.smiles
    
    
        graph, frags = get_frags(smiles)
        # get the atoms that belong to each fragment
        atoms_in_frags = get_atoms_in_frags(graph)
    
    
        for fid, frag_atoms in atoms_in_frags.items():
            data_copy = copy.deepcopy(item)
        
            atom_mask  = torch.zeros(data_copy['x_atoms'].shape[0], dtype=torch.int)
            atom_mask[ frag_atoms ] = 1
            data_copy['atom_mask'] = atom_mask
            data_copy['frag_atoms'] = frag_atoms
        
            # print(frag_atoms)
            # print(atom_mask)
            copies.append(data_copy)
    
        # make copies eqaul to the number of fragments in the mol
        # if i == 12:
        #     break
    return copies


from collections import defaultdict


def add_atom_weights(attribution, dataset):
    
    atom_weights = defaultdict()
    frag_atoms_list = defaultdict()
    for i in range(len(attribution)):
    
        w = attribution[i].item()
        frag_atoms = dataset[i]['frag_atoms']
        frag_atoms_list[i] = frag_atoms
        # wl = w.numpy().tolist()*len(frag_atoms)
    
        for i in frag_atoms:
            atom_weights[i] = w 

    return atom_weights, frag_atoms_list



from collections import defaultdict
from rdkit.Chem.Draw import rdMolDraw2D
import io
from PIL import Image

WIDTH = 400
HEIGHT = 300

def highlight_atoms(atom_weights, mol):
    
    highlightatoms = defaultdict(list)
    highlightbonds = defaultdict(list)
    bond_widths = defaultdict(list)
    atom_ids=[]
    arads = {}
    # for a in mol.GetAtoms():

    cmap_name='seismic_r'
    # cmap_name='seismic'
    # cmap_name='seismic_l'
    # mol = Chem.MolFromSmiles(smiles)
    cmap = cm.get_cmap(cmap_name, 10)
    norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
    plt_colors = cm.ScalarMappable(norm=norm, cmap=cmap)
    
    for i in range(len(atom_weights)):
        # for j in frag.fragments[i].atom_indices:
            # highlightatoms[j].append(colors[i])
        # highlightatoms[i].append(cm.seismic(atom_weights[i]))
        # if i in frag_atoms:
        #     highlightatoms[i].append(colors[1])
        highlightatoms[i] = [plt_colors.to_rgba(float(atom_weights[i]))]  #####
        # highlightatoms[i].append(cm.coolwarm(atom_weights[i]))
        # atom_ids.append(i)
        # arads[i] = 0.3
    
    
    highlight_atom_map = dict(highlightatoms)
    d2d = rdMolDraw2D.MolDraw2DCairo(width=WIDTH, height=HEIGHT)
    d2d.DrawMoleculeWithHighlights(mol=mol,legend="", highlight_atom_map=highlight_atom_map,
                                # highlight_bond_map={},
                                highlight_bond_map={},
                                highlight_radii={},highlight_linewidth_multipliers={})
    
    d2d.FinishDrawing()
    bio = io.BytesIO(d2d.GetDrawingText())
    png = Image.open(bio)

    return png



def get_attr_image(smiles, model_config, chkpt_path):

    dataset = FinetuneData(target_name='log_sol', data_type='exp1s')

    train = pd.DataFrame.from_dict({'smiles': [smiles], 'log_sol': [1.]})
    ds = dataset.get_ft_dataset(train)
    ds = extract_data(ds)

    dataset = create_data(ds)
    # model_config = '../fragnet/fragnet/exps/ft/pnnl_set2/fragnet_exp1s_h4pt4_noopt/config_exp100.yaml'
    # chkpt_path = '../fragnet/fragnet/exps/ft/pnnl_set2/fragnet_exp1s_h4pt4_noopt/ft_100.pt'

    # smiles = dataset[0].smiles
    pred_no_mask, pred_mask, true = get_predictions(dataset, model_config, chkpt_path)
    attribution = (pred_no_mask - pred_mask).numpy().ravel()
    atom_weights, frag_atoms = add_atom_weights(attribution, dataset)

    mol = get_mol(smiles)
    png = highlight_atoms(atom_weights, mol)

    return png