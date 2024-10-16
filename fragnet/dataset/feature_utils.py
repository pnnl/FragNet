import numpy as np
from rdkit.Chem import AllChem
import torch
import random
import os
from rdkit.Chem import rdDepictor
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import Chem

def get_symbols(df):
    
    s=[]
    for i in df.smiles:
        mol=Chem.MolFromSmiles(i)
        atoms = mol.GetAtoms()
        s +=[a.GetSymbol() for a in atoms]
    return set(s)


symbols = ['Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S']
symb_to_id = {v:k for k,v in enumerate(symbols)}
deg_to_id = {v:k for k,v in enumerate([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])}
valence_to_id = {v:k for k,v in enumerate([0, 1, 2, 3, 4, 5, 6])}
charge_to_id =  {v:k for k,v in enumerate([-3, -2, -1, 0, 1, 2, 3])}
nH_to_id =  {v:k for k,v in enumerate([0, 1, 2, 3, 4])}
radical_e_to_id =  {v:k for k,v in enumerate([0, 1, 2])}
arom_to_id =  {v:k for k,v in enumerate([0, 1])}
inring_to_id =  {v:k for k,v in enumerate([0, 1])}



hyb_to_id  = {v:k for k,v in enumerate([Chem.rdchem.HybridizationType.UNSPECIFIED,
                                        Chem.rdchem.HybridizationType.S,
                                        Chem.rdchem.HybridizationType.SP, 
                                        Chem.rdchem.HybridizationType.SP2,
                                        Chem.rdchem.HybridizationType.SP3, 
                                        Chem.rdchem.HybridizationType.SP3D,
                                        Chem.rdchem.HybridizationType.SP3D2])}

btype_to_id = {v:k for k,v in enumerate(['None', Chem.rdchem.BondType.SINGLE,
                                        Chem.rdchem.BondType.DOUBLE,
                                        Chem.rdchem.BondType.TRIPLE,
                                        Chem.rdchem.BondType.AROMATIC])} # reserve 0 for self edge bondtype

stero_to_id = {v:k for k,v in enumerate([Chem.rdchem.BondStereo.STEREONONE,
                                        Chem.rdchem.BondStereo.STEREOANY,
                                        Chem.rdchem.BondStereo.STEREOZ,
                                        Chem.rdchem.BondStereo.STEREOE])}

bonddir_to_id = {v:k for k,v in enumerate([Chem.rdchem.BondDir.NONE,
                                        Chem.rdchem.BondDir.BEGINWEDGE,
                                        Chem.rdchem.BondDir.BEGINDASH,
                                        Chem.rdchem.BondDir.ENDDOWNRIGHT,
                                        Chem.rdchem.BondDir.ENDUPRIGHT])}

conj_to_id = {v:k for k,v in enumerate([0, 1])}
inring_to_id = {v:k for k,v in enumerate([0, 1])}


def get_label(value, dict):
    if value in dict:
        return dict[value]
    else: 
        return len(dict)



def get_atom_features(atom):
    

    atom_token = get_label(atom.GetSymbol(), symb_to_id)        
    degree = get_label(atom.GetDegree(), deg_to_id)
    implvalence = get_label(atom.GetImplicitValence(), valence_to_id)
    nradelec = get_label(atom.GetNumRadicalElectrons(), radical_e_to_id)
    hyb = get_label(atom.GetHybridization(), hyb_to_id)
    arom = get_label(atom.GetIsAromatic(), arom_to_id)
    ring = get_label(atom.IsInRing(), inring_to_id)
    
    atom_feature = [atom_token, degree, implvalence, nradelec, hyb, arom, ring]
    
    return atom_feature

def get_bond_features(bond):
    

    btype = get_label(bond.GetBondType(), btype_to_id)
    stereo = get_label(bond.GetStereo(), stero_to_id)
    bdirec = get_label(bond.GetBondDir(), bonddir_to_id)
    conj = get_label(bond.GetIsConjugated(), conj_to_id)
    inring = get_label(bond.IsInRing(), inring_to_id)
 
  
    return [btype, stereo, bdirec, conj, inring]

def get_atom_and_bond_features_atom_graph(mol):
    
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    
    atom_features = [get_atom_features(a) for a in atoms]
    
    res, ea = [[],[]], []
    for bond in bonds:
        res[0] += [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        res[1] += [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]
        edge_attr_ = get_bond_features(bond)
        ea.append(edge_attr_)
        ea.append(edge_attr_)
        
    return atom_features, res, ea
            



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))
 
def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))
 
def get_intervals(l):
    """For list of lists, gets the cumulative products of the lengths"""
    intervals = len(l) * [0]
    # Initalize with 1
    intervals[0] = 1
    for k in range(1, len(l)):
        intervals[k] = (len(l[k]) + 1) * intervals[k - 1]
    return intervals
 
def safe_index(l, e):
    """Gets the index of e in l, providing an index of len(l) if not found"""
    try:
        return l.index(e)
    except:
        return len(l)

def best_fit_slope_and_intercept(xs,ys):
    m = (((np.mean(xs)*np.mean(ys)) - np.mean(xs*ys)) /
         ((np.mean(xs)*np.mean(xs)) - np.mean(xs*xs)))
    
    b = np.mean(ys) - m*np.mean(xs)
    
    return m, b

possible_atom_list = [
    'C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Mg', 'Na', 'Br', 'Fe', 'Ca', 'Cu',
    'Mc', 'Pd', 'Pb', 'K', 'I', 'Al', 'Ni', 'Mn'
]
possible_numH_list = [0, 1, 2, 3, 4]
possible_valence_list = [0, 1, 2, 3, 4, 5, 6]
possible_formal_charge_list = [-3, -2, -1, 0, 1, 2, 3]
possible_hybridization_list = [
    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2
]
possible_number_radical_e_list = [0, 1, 2]
possible_chirality_list = ['R', 'S']
 
reference_lists = [
    possible_atom_list, possible_numH_list, possible_valence_list,
    possible_formal_charge_list, possible_number_radical_e_list,
    possible_hybridization_list, possible_chirality_list
]
 
intervals = get_intervals(reference_lists)


def get_feature_list(atom):
    features = 6 * [0]
    features[0] = safe_index(possible_atom_list, atom.GetSymbol())
    features[1] = safe_index(possible_numH_list, atom.GetTotalNumHs())
    features[2] = safe_index(possible_valence_list, atom.GetImplicitValence())
    features[3] = safe_index(possible_formal_charge_list, atom.GetFormalCharge())
    features[4] = safe_index(possible_number_radical_e_list,
                           atom.GetNumRadicalElectrons())
    features[5] = safe_index(possible_hybridization_list, atom.GetHybridization())
    return features
 
def features_to_id(features, intervals):
    """Convert list of features into index using spacings provided in intervals"""
    id = 0
    for k in range(len(intervals)):
        id += features[k] * intervals[k]

        # Allow 0 index to correspond to null molecule 1
        id = id + 1
    return id

def id_to_features(id, intervals):
    features = 6 * [0]

    # Correct for null
    id -= 1

    for k in range(0, 6 - 1):
        # print(6-k-1, id)
        features[6 - k - 1] = id // intervals[6 - k - 1]
        id -= features[6 - k - 1] * intervals[6 - k - 1]
        # Correct for last one
        features[0] = id
    return features

def atom_to_id(atom):
    """Return a unique id corresponding to the atom type"""
    features = get_feature_list(atom)
    return features_to_id(features, intervals)



def get_bond_pair(mol, add_self_loops=False):
    bonds = mol.GetBonds()
    res = [[],[]]
    for bond in bonds:
        res[0] += [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        res[1] += [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]
        
        
    if add_self_loops:
        res[0]+= list(range(mol.GetNumAtoms()))
        res[1]+= list(range(mol.GetNumAtoms()))
        
    return res


def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol

def get_indices(mol):
    
    indices=[]
    atoms = mol.GetAtoms()
    for a in atoms:
        indices.append(a.GetAtomMapNum())
    return indices

def flatten_list(_2d_list):
    flat_list = []
    # Iterate through the outer list
    for element in _2d_list:
        if type(element) is list:
            # If the element is of type list, iterate through the sublist
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list

from rdkit.Chem import MACCSkeys
def get_hyperedge_indices(mol):
    
    mol = mol_with_atom_index(mol)
    
    core = MurckoScaffold.GetScaffoldForMol(mol)
    rm = AllChem.DeleteSubstructs(mol, core)
    rm = Chem.MolToSmiles(rm)

    fg_smiles = rm.split('.')
    fgs = [Chem.MolFromSmiles(s) for s in fg_smiles]
    
    hedges = [get_indices(core)] + [get_indices(fg) for fg in fgs]
    
    hedge_attr = [MACCSkeys.GenMACCSKeys(x) for x in [core]+fgs ]
    hedge_w = len(fgs)*[1]+[1]
    
    i=0
    hids=[]
    for h in hedges:
        hids.extend(len(h)*[i])
        i+=1
    
    hedges = flatten_list(hedges)
    hyperedge_index = [hedges, hids]
    
    return hyperedge_index, hedge_attr, hedge_w


pt = Chem.GetPeriodicTable()
def get_xyz(path):
    f = open(path, 'r')
    f = f.readlines()
    atoms = f[2:]

    natoms = int(f[0].strip('\n'))

    atom_z = [atoms[i].split()[0] for i in range(natoms)]
    pos = [[float(atoms[i].split()[1:][j]) for j in range(3)] for i in range(natoms)]
    return atom_z, pos


def create_minmol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol.GetNumConformers():
        rdDepictor.Compute2DCoords(mol)

    conf = mol.GetConformer()
    Chem.WedgeMolBonds(mol, conf)

    return mol
