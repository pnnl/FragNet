import itertools
from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem.rdmolfiles import MolFragmentToSmiles
import numpy as np
from rdkit.Chem import AllChem
from rdkit.Chem import rdDistGeom
from rdkit.Chem import rdMolAlign
import collections
from collections import defaultdict
from .utils import remove_bond
from rdkit.Chem.Scaffolds import MurckoScaffold


def find_murcko_link_bond(mol):
    core = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_index = mol.GetSubstructMatch(core)
    link_bond_list = []
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        link_score = 0
        if u in scaffold_index:
            link_score += 1
        if v in scaffold_index:
            link_score += 1
        if link_score == 1:
            link_bond_list.append([u, v])
    return link_bond_list


def smi2_2Dcoords(smi):
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    return mol


def get_3Dcoords(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = AllChem.AddHs(mol)
    seed = 42
    res = AllChem.EmbedMolecule(mol, randomSeed=seed)
    try:
        if res == 0:
            try:
                AllChem.MMFFOptimizeMolecule(
                    mol
                )  # some conformer can not use MMFF optimize
                coordinates = mol.GetConformer()
            except:
                print("Failed to generate 3D, replace with 2D")
                mol = smi2_2Dcoords(smiles)

        elif res == -1:
            mol = Chem.MolFromSmiles(smiles)
            AllChem.EmbedMolecule(mol, maxAttempts=5000, randomSeed=seed)
            mol = AllChem.AddHs(mol, addCoords=True)
            try:
                AllChem.MMFFOptimizeMolecule(
                    mol
                )  # some conformer can not use MMFF optimize
                coordinates = mol.GetConformer()
            except:
                print("Failed to generate 3D, replace with 2D")
                mol = smi2_2Dcoords(smiles)
    except:
        print("Failed to generate 3D, replace with 2D")
        mol = smi2_2Dcoords(smiles)

    return mol


def get_3Dcoords2(smiles, numconf=1, maxiters=200):
    prunermsthresh = 0.1

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    refmol = Chem.AddHs(Chem.Mol(mol))
    param = rdDistGeom.ETKDGv2()
    param.pruneRmsThresh = prunermsthresh

    cids = rdDistGeom.EmbedMultipleConfs(mol, numconf, param)
    mp = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
    try:
        o = AllChem.MMFFOptimizeMoleculeConfs(
            mol, numThreads=0, mmffVariant="MMFF94s", maxIters=maxiters
        )
    except:
        return None

    if not o or len(o) < 1:
        return None

    res = []
    for i in range(len(cids)):
        if o[i][0] == 0:

            cid = cids[i]
            ff = AllChem.MMFFGetMoleculeForceField(mol, mp, confId=cid)
            e = ff.CalcEnergy()
            res.append((cid, e))

        else:
            return None

    return mol, res


class Fragment:
    def __init__(self, graph, atom_indices, FragIdx=0):

        self.FragIdx = FragIdx
        self.graph = graph
        atom_indices_set = set(atom_indices)
        bond_indices = []
        for bond in graph.mol.GetBonds():
            atom_idx1, atom_idx2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if atom_idx1 in atom_indices_set and atom_idx2 in atom_indices_set:
                bond_indices.append(bond.GetIdx())

        self.atom_indices = atom_indices
        self.bond_indices = tuple(bond_indices)

        frag_rdmol = Chem.MolFromSmiles(
            MolFragmentToSmiles(graph.mol, atom_indices), sanitize=False
        )
        self.frag = frag_rdmol

        self.neighbors = []
        self.connections = []

    def add_connection(self, neighbor_unit, connection):
        self.neighbors.append(neighbor_unit)
        self.connections.append(connection)


class EmptyBond:
    def __init__(self):
        pass

    def GetIsConjugated(self):
        return False

    def GetBondDir(self):
        return Chem.rdchem.BondDir.NONE

    def IsInRing(self):
        return False

    def GetStereo(self):
        return "STEREONONE"


class Connection:
    def __init__(
        self, frag1, frag2, bond_atom_id1, bond_atom_id2, bond_index, bond_type, bond
    ):

        frag1.add_connection(frag2, self)
        frag2.add_connection(frag1, self)

        self.frags = (frag1, frag2)
        self.atom_indices = (bond_atom_id1, bond_atom_id2)
        self.bond_id = bond_index
        self.bond_type = bond_type
        self.BeginFragIdx = frag1.FragIdx
        self.EndFragIdx = frag2.FragIdx
        self.bond = bond


class FragmentedMol:
    def __init__(self, mol, conf, frag_type="brics"):

        Chem.WedgeMolBonds(mol, conf)
        self.mol = mol
        # get the brics bonds
        if frag_type == "brics":
            frag_bonds = list(BRICS.FindBRICSBonds(mol))
            frag_bonds = [b[0] for b in frag_bonds]
        elif frag_type == "murcko":
            # print('using murcko')
            frag_bonds = find_murcko_link_bond(mol)

        rwmol = Chem.RWMol(mol)

        for atom_idx1, atom_idx2 in frag_bonds:
            remove_bond(rwmol, atom_idx1, atom_idx2)

        # get the fragments
        broken_mol = rwmol.GetMol()

        # get the atom indices of the fragments
        atomMap = Chem.GetMolFrags(broken_mol)

        # get the fragment objects
        fragments = []
        for i, atom_indices in enumerate(atomMap):
            fragments.append(Fragment(self, atom_indices, FragIdx=i))

        self.fragments = fragments
        self.atom_to_frag_id = self.get_atom_to_frag_id()

        # atom_id to fragment dict
        fragment_map = {}
        for frag in fragments:
            for atom_id in frag.atom_indices:
                fragment_map[atom_id] = frag

        connections = []
        # for bric_bond in brics_bonds:
        for frag_bond in frag_bonds:
            (atom_id1, atom_id2) = frag_bond
            bond = mol.GetBondBetweenAtoms(atom_id1, atom_id2)
            bond_type, bond_index = bond.GetBondType(), bond.GetIdx()

            frag1 = fragment_map[atom_id1]
            frag2 = fragment_map[atom_id2]

            # add the connections
            # store the connection information
            connection = Connection(
                frag1, frag2, atom_id1, atom_id2, bond_index, bond_type, bond
            )
            connections.append(connection)

        # if there are no fragments, the fragment is the whole mol itself
        # make a connection with itself
        if len(connections) == 0 and len(fragments) == 1:
            frag1 = fragments[0]
            frag2 = fragments[0]
            bond = EmptyBond()
            connections = [Connection(frag1, frag2, None, None, None, "self_cn", bond)]

        if len(Chem.GetMolFrags(self.mol)) > 1:
            sg_frags = self.get_atoms_in_molfrags()
            bond = EmptyBond()
            new_conections = self.add_connections_bw_molfrags(sg_frags, bond)
            connections = connections + new_conections

        self.connections = tuple(connections)

    def get_atom_to_frag_id(self):

        atom_to_frag_id = {}
        for i, f in enumerate(self.fragments):
            for a in f.atom_indices:
                atom_to_frag_id[a] = i

        atom_to_frag_id = collections.OrderedDict(sorted(atom_to_frag_id.items()))

        return atom_to_frag_id

    def get_atoms_in_molfrags(self):

        mol_frags = Chem.GetMolFrags(self.mol)
        nmol_frags = len(mol_frags)

        sg_frags = defaultdict(list)

        for i in range(nmol_frags):
            sg = set(mol_frags[i])

            for frag in self.fragments:

                frag_atoms = set(frag.atom_indices)
                if frag_atoms.issubset(sg):
                    sg_frags[i].append(frag)

        return sg_frags

    def add_connections_bw_molfrags(self, sg_frags, bond):

        new_conections = []
        for i in range(len(sg_frags)):
            set_i = sg_frags[i]
            for j in range(i + 1, len(sg_frags)):
                set_j = sg_frags[j]

                for fragi in set_i:
                    fragi_cnx = fragi.connections
                    for fragj in set_j:
                        fragj_cnx = fragj.connections

                        cnx_count = 0
                        fragi_cnx_ids = []
                        fragj_cnx_ids = []
                        for ic in fragi_cnx:
                            fragi_cnx_ids.append(
                                sorted((ic.BeginFragIdx, ic.EndFragIdx))
                            )

                        if sorted((fragi.FragIdx, fragj.FragIdx)) not in fragi_cnx_ids:
                            # print('updating')
                            connection = Connection(
                                fragi, fragj, None, None, None, "iso_cn3", bond
                            )
                            new_conections.append(connection)

        return new_conections
