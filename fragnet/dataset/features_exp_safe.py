import numpy as np
from rdkit import Chem
import multiprocessing
from torch_geometric.data import Data
import torch.nn as nn
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn import BatchNorm1d
from torch.utils.data import Dataset
from torch_geometric.nn import GCNConv
from torch_geometric.nn import ChebConv
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.data import DataLoader
from torch_scatter import scatter_mean
import pandas as pd
from sklearn.model_selection import train_test_split
from random import randrange
import itertools
from torch_geometric.nn import EdgeConv
import random
import os
import time
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

import pickle, gzip
from rdkit.Chem import rdDepictor
from rdkit.Chem.Scaffolds import MurckoScaffold

# import config
from rdkit import Chem


def get_symbols(df):

    s = []
    for i in df.smiles:
        mol = Chem.MolFromSmiles(i)
        atoms = mol.GetAtoms()
        s += [a.GetSymbol() for a in atoms]
    return set(s)


# def get_feature_dicts(df):

# mols = [Chem.MolFromSmiles(i) for i in df.smiles]
# symbols = [[a.GetSymbol() for a in mol.GetAtoms()] for mol in mols]
# symbols = sum(symbols, [])
# symbols = set(symbols)
symbols = ["Br", "C", "Cl", "F", "I", "N", "O", "P", "S"]
symb_to_id = {v: k for k, v in enumerate(symbols)}
deg_to_id = {v: k for k, v in enumerate([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])}
valence_to_id = {v: k for k, v in enumerate([0, 1, 2, 3, 4, 5, 6])}
charge_to_id = {v: k for k, v in enumerate([-3, -2, -1, 0, 1, 2, 3])}
nH_to_id = {v: k for k, v in enumerate([0, 1, 2, 3, 4])}
radical_e_to_id = {v: k for k, v in enumerate([0, 1, 2])}
arom_to_id = {v: k for k, v in enumerate([0, 1])}
inring_to_id = {v: k for k, v in enumerate([0, 1])}


hyb_to_id = {
    v: k
    for k, v in enumerate(
        [
            Chem.rdchem.HybridizationType.UNSPECIFIED,
            Chem.rdchem.HybridizationType.S,
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
        ]
    )
}

btype_to_id = {
    v: k
    for k, v in enumerate(
        [
            "None",
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC,
        ]
    )
}  # reserve 0 for self edge bondtype

stero_to_id = {
    v: k
    for k, v in enumerate(
        [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
        ]
    )
}

bonddir_to_id = {
    v: k
    for k, v in enumerate(
        [
            Chem.rdchem.BondDir.NONE,
            Chem.rdchem.BondDir.BEGINWEDGE,
            Chem.rdchem.BondDir.BEGINDASH,
            Chem.rdchem.BondDir.ENDDOWNRIGHT,
            Chem.rdchem.BondDir.ENDUPRIGHT,
        ]
    )
}

conj_to_id = {v: k for k, v in enumerate([0, 1])}
inring_to_id = {v: k for k, v in enumerate([0, 1])}

# return symb_to_id, hyb_to_id, btype_to_id, stero_to_id, bonddir_to_id


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

    res, ea = [[], []], []
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
    os.environ["PYTHONHASHSEED"] = str(seed)


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


def best_fit_slope_and_intercept(xs, ys):
    m = ((np.mean(xs) * np.mean(ys)) - np.mean(xs * ys)) / (
        (np.mean(xs) * np.mean(xs)) - np.mean(xs * xs)
    )

    b = np.mean(ys) - m * np.mean(xs)

    return m, b


possible_atom_list = [
    "C",
    "N",
    "O",
    "S",
    "F",
    "P",
    "Cl",
    "Mg",
    "Na",
    "Br",
    "Fe",
    "Ca",
    "Cu",
    "Mc",
    "Pd",
    "Pb",
    "K",
    "I",
    "Al",
    "Ni",
    "Mn",
]
possible_numH_list = [0, 1, 2, 3, 4]
possible_valence_list = [0, 1, 2, 3, 4, 5, 6]
possible_formal_charge_list = [-3, -2, -1, 0, 1, 2, 3]
possible_hybridization_list = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
possible_number_radical_e_list = [0, 1, 2]
possible_chirality_list = ["R", "S"]

reference_lists = [
    possible_atom_list,
    possible_numH_list,
    possible_valence_list,
    possible_formal_charge_list,
    possible_number_radical_e_list,
    possible_hybridization_list,
    possible_chirality_list,
]

intervals = get_intervals(reference_lists)


def get_feature_list(atom):
    features = 6 * [0]
    features[0] = safe_index(possible_atom_list, atom.GetSymbol())
    features[1] = safe_index(possible_numH_list, atom.GetTotalNumHs())
    features[2] = safe_index(possible_valence_list, atom.GetImplicitValence())
    features[3] = safe_index(possible_formal_charge_list, atom.GetFormalCharge())
    features[4] = safe_index(
        possible_number_radical_e_list, atom.GetNumRadicalElectrons()
    )
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


# atom_list_one_hot = ['Ag','Al','As','B','Br','C','Ca','Cd','Cl','Cu','F',
#                      'Fe','Ge','H','Hg','I','K','Li','Mg','Mn','N','Na',
#                      'O','P','Pb','Pt','S','Se','Si','Sn','Sr','Tl','Zn','Unknown']
# atom_list_one_hot = ['Br','C','Cl','F','H','I','K','N','Na','O','P','S','Unknown']


# degree 1hot, valence 1hot
# charge 1hot
# rad_elec
def atom_features_one_hot(
    atom, bool_id_feat=False, explicit_H=False, use_chirality=False, angle_f=False
):
    atom_list_one_hot = list(range(1, 119))
    if bool_id_feat:
        return np.array([atom_to_id(atom)])
    else:

        # atom_type = [atom.GetAtomicNum()]
        atom_type = one_of_k_encoding_unk(atom.GetAtomicNum(), atom_list_one_hot)
        # degree = [atom.GetDegree()]
        # valence = [atom.GetImplicitValence()]
        degree = one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        # degree = one_of_k_encoding(atom.GetTotalDegree(),[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        valence = one_of_k_encoding_unk(
            atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]
        )
        # charge = [atom.GetFormalCharge()]
        charge = one_of_k_encoding_unk(
            atom.GetFormalCharge(), [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
        )
        rad_elec = one_of_k_encoding_unk(atom.GetNumRadicalElectrons(), [0, 1, 2, 3, 4])
        # rad_elec = [atom.GetNumRadicalElectrons()]

        hyb = one_of_k_encoding_unk(
            atom.GetHybridization(),
            [
                Chem.rdchem.HybridizationType.S,
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2,
                Chem.rdchem.HybridizationType.UNSPECIFIED,
            ],
        )
        # arom = [atom.GetIsAromatic()]
        arom = one_of_k_encoding(atom.GetIsAromatic(), [False, True])
        # atom_ring = [atom.IsInRing()]
        atom_ring = one_of_k_encoding(atom.IsInRing(), [False, True])
        # mass = [atom.GetMass()]
        numhs = [atom.GetTotalNumHs()]
        # numhs = one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8])

        chiral = one_of_k_encoding_unk(
            atom.GetChiralTag(),
            [
                Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
                # Chem.rdchem.ChiralType.CHI_OTHER,
                # Chem.rdchem.ChiralType.CHI_TETRAHEDRAL,
                # Chem.rdchem.ChiralType.CHI_ALLENE,
                # Chem.rdchem.ChiralType.CHI_SQUAREPLANAR,
                # Chem.rdchem.ChiralType.CHI_TRIGONALBIPYRAMIDAL,
                # Chem.rdchem.ChiralType.CHI_OCTAHEDRAL
            ],
        )

        results = (
            atom_type
            + degree
            + valence
            + charge
            + rad_elec
            + hyb
            + arom
            + atom_ring
            + chiral
            + numhs
        )

        #         if angle_f:
        #             results = results + [math.cos(float(atom.GetProp('angle1'))),\
        #                                  math.cos(float(atom.GetProp('angle2'))),\
        #                                  math.cos(float(atom.GetProp('angle3')))]

        # if not explicit_H:
        #     results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
        #                                         [0, 1, 2, 3, 4])
        # if use_chirality:
        #     try:
        #         results = results + one_of_k_encoding_unk(atom.GetProp('_CIPCode'),
        #                                                   ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        #     except:
        #         results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]

        return np.array(results)


def bond_features_one_hot(bond, use_chirality=True):
    bt = bond.GetBondType()

    # begin = bond.GetBeginAtomIdx()
    # end = bond.GetEndAtomIdx()

    #     c = mol.GetConformer()

    #     Adding dihedral angle as a bond feature. Method 1.
    #     try:
    #         begin_nebr = get_bond_nebrs(mol=mol, c=c, begin_atom=1, end_atom=0)[1]
    #         end_nebr = get_bond_nebrs(mol=mol, c=c, begin_atom=0, end_atom=1)[1]

    #         pbond = [begin_nebr, begin, end, end_nebr]
    #         dhbond = Chem.rdMolTransforms.GetDihedralDeg(c, *pbond)
    #     except:
    #         dhbond = 0
    #     Adding dihedral angle as a bond feature. Method 1.

    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        # bond.GetIsConjugated(),
        # bond.IsInRing()
    ]

    conj = one_of_k_encoding(bond.GetIsConjugated(), [False, True])
    inring = one_of_k_encoding(bond.IsInRing(), [False, True])

    bond_feats = bond_feats + conj + inring

    if use_chirality:
        bond_feats = bond_feats + one_of_k_encoding_unk(
            str(bond.GetStereo()), ["STEREOANY", "STEREOZ", "STEREOE", "STEREONONE"]
        )

    bond_feats = bond_feats + one_of_k_encoding_unk(
        bond.GetBondDir(),
        [
            Chem.rdchem.BondDir.BEGINWEDGE,
            Chem.rdchem.BondDir.BEGINDASH,
            Chem.rdchem.BondDir.ENDDOWNRIGHT,
            Chem.rdchem.BondDir.ENDUPRIGHT,
            Chem.rdchem.BondDir.NONE,
        ],
    )

    #     bond_feats = bond_feats + [math.cos(dhbond)]
    return list(bond_feats)


def connection_features_one_hot(connection):

    bt = connection.bond_type

    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bt == "no_cn_have_cn",
        bt == "no_cn_no_cn",
        bt == "self_cn",
        bt == "iso_cn",
    ]
    return list(bond_feats)


def get_bond_pair(mol, add_self_loops=False):
    bonds = mol.GetBonds()
    res = [[], []]
    for bond in bonds:
        res[0] += [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        res[1] += [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]

    if add_self_loops:
        res[0] += list(range(mol.GetNumAtoms()))
        res[1] += list(range(mol.GetNumAtoms()))

    return res


def get_atom_and_bond_features_atom_graph_one_hot(mol, use_chirality):

    #     mol = Chem.MolFromSmiles(smiles)
    #     mol = create_minmol(smiles)

    add_self_loops = False

    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    edge_index = get_bond_pair(mol, add_self_loops)

    node_f = [atom_features_one_hot(atom) for atom in atoms]
    edge_attr = []
    for bond in bonds:
        edge_attr.append(bond_features_one_hot(bond, use_chirality=use_chirality))
        edge_attr.append(bond_features_one_hot(bond, use_chirality=use_chirality))

    if add_self_loops:
        self_loop_attr = np.zeros((mol.GetNumAtoms(), 12)).tolist()

        edge_attr = edge_attr + self_loop_attr

    # if len(edge_index[0]) ==0:
    #     edge_index = [[ i for i in range(len(atoms))], [ i for i in range(len(atoms))]]
    #     edge_attr = [[0]*12]

    return node_f, edge_index, edge_attr


def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol


def get_indices(mol):

    indices = []
    atoms = mol.GetAtoms()
    for a in atoms:
        #         print(a.GetAtomMapNum())
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

    fg_smiles = rm.split(".")
    fgs = [Chem.MolFromSmiles(s) for s in fg_smiles]

    hedges = [get_indices(core)] + [get_indices(fg) for fg in fgs]

    hedge_attr = [MACCSkeys.GenMACCSKeys(x) for x in [core] + fgs]
    hedge_w = len(fgs) * [1] + [1]

    # hedge_attr
    #     M_smiles = [Chem.MolToSmiles(core)]  + fg_smiles
    #     test = pd.DataFrame([[sm , 0] for sm in M_smiles], columns=['smiles', 'log_sol'])
    #     _, hedge_attr = model.predict(test, test, test)

    i = 0
    hids = []
    for h in hedges:
        hids.extend(len(h) * [i])
        i += 1

    hedges = flatten_list(hedges)
    hyperedge_index = [hedges, hids]

    return hyperedge_index, hedge_attr, hedge_w


# MOL CREATION
def create_mol(path):
    #     cas = dft.cas.values[i]

    #     path = xyz_dir + cas + ".xyz"
    #     path = xyz_dir + cas + "_noH.xyz"
    path = path

    Z_atoms, coordinates = get_xyz(path)
    Z_atoms = fix_Z(Z_atoms)
    mol = next(pybel.readfile("xyz", path))
    mol.write("sdf", "tmp.sdf", overwrite=True)
    file_name = "./tmp.sdf"
    #     file_name = "../../../sdf_nw/"+str(cas)+".sdf"
    suppl = Chem.SDMolSupplier(file_name, removeHs=False)
    mol = suppl[0]
    #     mol = Chem.AddHs(mol)
    atoms = mol.GetAtoms()
    #             print(i)

    #     print(len(coordinates))
    #     print(coordinates)
    coordinates = np.array(coordinates)
    for j in range(len(coordinates)):
        assert Z_atoms[j] == atoms[j].GetSymbol()
        mol.GetAtomWithIdx(j).SetDoubleProp("x", coordinates[j][0])
        mol.GetAtomWithIdx(j).SetDoubleProp("y", coordinates[j][1])
        mol.GetAtomWithIdx(j).SetDoubleProp("z", coordinates[j][2])

        # calculating dihedral for atom with index j
        #         dh1, dh2 = get_atom_dh(mol, j)
        #         mol.GetAtomWithIdx(j).SetDoubleProp('dh1', dh1 )
        #         mol.GetAtomWithIdx(j).SetDoubleProp('dh2', dh2 )

        # -----------------
        # calculating angle
        # -----------------
        b1 = j
        b1_xyz = coordinates[b1]

        dists = np.sum((coordinates - b1_xyz) ** 2, axis=1)
        dist_argsort_b1 = np.argsort(dists)
        #         dist_argsort_b1 = np.argsort(np.sum((coordinates - b1_xyz )**2, axis=1))

        if len(dist_argsort_b1) > 3:

            n1 = dist_argsort_b1[1]
            n2 = dist_argsort_b1[2]
            n3 = dist_argsort_b1[3]

            #         angle = get_angle(coordinates[n1], b1_xyz, coordinates[n2])
            try:
                angle1 = get_angle(coordinates[n1], b1_xyz, coordinates[n2])
            except:
                angle1 = 0

            try:
                angle2 = get_angle(coordinates[n1], b1_xyz, coordinates[n3])
            except:
                angle2 = 0

            try:
                angle3 = get_angle(coordinates[n2], b1_xyz, coordinates[n3])
            except:
                angle3 = 0

            angles = [angle1, angle2, angle3]
            angles.sort()

            maxd = dists[dist_argsort_b1[-1]]
            mind = dists[dist_argsort_b1[1]]

            #             dv =[]
            for iden, idist in enumerate(range(0, 40, 2)):
                d = dists[(dists >= idist) & (dists < idist + 2)]
                d = len(d)
                mol.GetAtomWithIdx(j).SetProp(f"atominl{iden + 1}", str(d))

        #             for i in range(1,21):

        else:
            angles = [0, 0, 0]
            maxd, mind = 0, 0

            for iden, idist in enumerate(range(0, 40, 2)):
                mol.GetAtomWithIdx(j).SetProp(f"atominl{iden + 1}", str(0))

        #         print(angle)

        #         angles = sorted(angles)
        mol.GetAtomWithIdx(j).SetProp("angle1", str(angles[0]))
        mol.GetAtomWithIdx(j).SetProp("angle2", str(angles[1]))
        mol.GetAtomWithIdx(j).SetProp("angle3", str(angles[2]))

        mol.GetAtomWithIdx(j).SetProp("maxd", str(maxd))
        mol.GetAtomWithIdx(j).SetProp("mind", str(mind))

    #         if not np.isnan(angle):
    #             mol.GetAtomWithIdx(j).SetProp('angle', str(angle) )
    #         else:
    #             mol.GetAtomWithIdx(j).SetProp('angle', str(0) )

    #     mol = Chem.RemoveHs(mol)
    return mol


pt = Chem.GetPeriodicTable()


def get_xyz(path):
    f = open(path, "r")
    f = f.readlines()
    atoms = f[2:]

    natoms = int(f[0].strip("\n"))

    atom_z = [atoms[i].split()[0] for i in range(natoms)]
    pos = [[float(atoms[i].split()[1:][j]) for j in range(3)] for i in range(natoms)]
    #     pos = [[float(atoms[i].split()[1:][j]) for j in range(3)] for i in range(natoms) if atom_z[i] !='H']
    #     atom_z = [atom_z[i] for i in range(natoms) if atom_z[i] !='H']

    return atom_z, pos


def create_minmol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol.GetNumConformers():
        rdDepictor.Compute2DCoords(mol)

    conf = mol.GetConformer()
    Chem.WedgeMolBonds(mol, conf)

    return mol


# MOL CREATION
