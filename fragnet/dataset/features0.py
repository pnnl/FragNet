import numpy as np
from rdkit import Chem
from feature_utils import one_of_k_encoding, one_of_k_encoding_unk
from feature_utils import get_bond_pair

class FeaturesEXP:

    def __init__(self):
        self.atom_list_one_hot =  ['Br','C','Cl','F','H','I','K','N','Na','O','P','S','Unknown']
        self.use_bond_chirality=False


    def get_atom_and_bond_features_atom_graph_one_hot(self, mol, use_chirality):
    
        add_self_loops=False

        atoms = mol.GetAtoms()
        bonds = mol.GetBonds()
        edge_index = get_bond_pair(mol, add_self_loops)

        
        node_f = [self.atom_features_one_hot(atom) for atom in atoms]    
        edge_attr=[]
        for bond in bonds:
            edge_attr.append(self.bond_features_one_hot(bond, use_chirality=use_chirality))
            edge_attr.append(self.bond_features_one_hot(bond, use_chirality=use_chirality))
            
            
        if add_self_loops:
            self_loop_attr = np.zeros(( mol.GetNumAtoms(),12)).tolist()
        
            edge_attr = edge_attr + self_loop_attr
            
            
        return node_f, edge_index, edge_attr

    def atom_features_one_hot(self, atom, bool_id_feat=False, explicit_H=False, use_chirality=False, angle_f=False):
        
            
        atom_type = one_of_k_encoding_unk(atom.GetSymbol(), self.atom_list_one_hot)
        degree = one_of_k_encoding(atom.GetDegree(),[0, 1, 2, 3, 4, 5, 6])
        valence = one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6])
        charge = [atom.GetFormalCharge()]
        rad_elec = [atom.GetNumRadicalElectrons()]
        hyb = one_of_k_encoding_unk( atom.GetHybridization(), [
            Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            Chem.rdchem.HybridizationType.UNSPECIFIED
            ]
                                )
        arom = [atom.GetIsAromatic()]
        atom_ring = [atom.IsInRing()]
        numhs = [atom.GetTotalNumHs()]

        chiral = one_of_k_encoding_unk(atom.GetChiralTag(),[Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                                                        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                                                        Chem.rdchem.ChiralType.CHI_UNSPECIFIED
                                                        ]
                                                        )


        results = atom_type+degree+valence+charge+rad_elec+hyb+arom+atom_ring+numhs

        if use_chirality:
            try:
                results = results + one_of_k_encoding_unk(atom.GetProp('_CIPCode'),
                                                        ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
            except:
                results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]

        return np.array(results)



        
    def bond_features_one_hot(self, bond, use_chirality=True):
        bt = bond.GetBondType()

        bond_feats = [
            bt == Chem.rdchem.BondType.SINGLE, 
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE, 
            bt == Chem.rdchem.BondType.AROMATIC,
            bond.GetIsConjugated(),
            bond.IsInRing()
        ]

        if use_chirality:
            bond_feats = bond_feats + one_of_k_encoding_unk(str(bond.GetStereo()),
                                                            ["STEREOANY", "STEREOZ", "STEREOE", "STEREONONE"])

        bond_feats = bond_feats + one_of_k_encoding_unk(bond.GetBondDir(), [
                Chem.rdchem.BondDir.BEGINWEDGE,
                Chem.rdchem.BondDir.BEGINDASH,
                Chem.rdchem.BondDir.ENDDOWNRIGHT,
                Chem.rdchem.BondDir.ENDUPRIGHT,
                Chem.rdchem.BondDir.NONE
            ])

        return list(bond_feats)


    def connection_features_one_hot(self, connection):
        
        bt = connection.bond_type

        bond_feats = [
            bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
            bt == 'self_cn', bt=='iso_cn3'

    
        ]
        return list(bond_feats)