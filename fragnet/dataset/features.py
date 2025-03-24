import numpy as np
from rdkit import Chem
from .feature_utils import one_of_k_encoding, one_of_k_encoding_unk
from .feature_utils import get_bond_pair


class FeaturesEXP:

    """
    Class for creating initial atom, bond and connection features.
    
    """
    def __init__(self, add_connection_chrl=False):
        self.atom_list_one_hot = list(range(1, 119))
        self.use_bond_chirality = True
        self.add_connection_chrl = add_connection_chrl

    def get_atom_and_bond_features_atom_graph_one_hot(self, mol, use_chirality):

        add_self_loops = False

        atoms = mol.GetAtoms()
        bonds = mol.GetBonds()
        edge_index = get_bond_pair(mol, add_self_loops)

        node_f = [self.atom_features_one_hot(atom) for atom in atoms]
        edge_attr = []
        for bond in bonds:
            edge_attr.append(
                self.bond_features_one_hot(bond, use_chirality=use_chirality)
            )
            edge_attr.append(
                self.bond_features_one_hot(bond, use_chirality=use_chirality)
            )

        return node_f, edge_index, edge_attr

    def atom_features_one_hot(
        self, atom, explicit_H=False, use_chirality=False, angle_f=False
    ):

        atom_type = one_of_k_encoding_unk(atom.GetAtomicNum(), self.atom_list_one_hot)
        degree = one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        valence = one_of_k_encoding_unk(
            atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]
        )

        charge = one_of_k_encoding_unk(
            atom.GetFormalCharge(), [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
        )
        rad_elec = one_of_k_encoding_unk(atom.GetNumRadicalElectrons(), [0, 1, 2, 3, 4])

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

        arom = one_of_k_encoding(atom.GetIsAromatic(), [False, True])
        atom_ring = one_of_k_encoding(atom.IsInRing(), [False, True])
        numhs = [atom.GetTotalNumHs()]

        chiral = one_of_k_encoding_unk(
            atom.GetChiralTag(),
            [
                Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
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

        return np.array(results)

    def bond_features_one_hot(self, bond, use_chirality=True):
        bt = bond.GetBondType()

        bond_feats = [
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
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
        return list(bond_feats)

    def connection_features_one_hot(self, connection):

        bond = connection.bond
        bt = connection.bond_type

        bond_feats = [
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            bt == "self_cn",
            bt == "iso_cn3",
        ]

        if self.add_connection_chrl:

            conj = one_of_k_encoding(bond.GetIsConjugated(), [False, True])
            inring = one_of_k_encoding(bond.IsInRing(), [False, True])

            bond_feats = bond_feats + conj + inring
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

        return list(bond_feats)
