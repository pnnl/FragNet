import numpy as np
import pickle
import pandas as pd
import os
import lmdb
from rdkit import Chem
from tqdm import tqdm


def remove_non_mols(ds):

    keep_smiles = []
    keep_ids = []
    if isinstance(ds, pd.DataFrame):
        smiles_list = ds.smiles.values
    else:
        smiles_list = [i.smiles for i in ds]

    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            keep_ids.append(i)

    if isinstance(ds, pd.DataFrame):
        df2 = ds.loc[keep_ids, :]
        df2.reset_index(drop=True, inplace=True)
    else:
        df2 = ds.index_select(keep_ids)

    return df2


def extract_data(ds):
    res = []
    for i, data in tqdm(enumerate(ds)):
        if data:
            res.append(data)
    return res


def save_datasets(ds, save_path):
    with open(f"{save_path}.pkl", "wb") as f:
        pickle.dump(ds, f)


def remove_frags():

    frags = [len(Chem.GetMolFrags(Chem.MolFromSmiles(sm))) for sm in train.smiles]
    train["mfrags"] = frags

    frags = [len(Chem.GetMolFrags(Chem.MolFromSmiles(sm))) for sm in val.smiles]
    val["mfrags"] = frags

    train = train[train.mfrags == 1]
    val = val[val.mfrags == 1]

    # train.shape

    train.reset_index(drop=True, inplace=True)
    val.reset_index(drop=True, inplace=True)


def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol


# TODO: test this function
def remove_bond(rwmol, idx1, idx2):
    rwmol.RemoveBond(idx1, idx2)
    for idx in [idx1, idx2]:
        atom = rwmol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == "N" and atom.GetIsAromatic() is True:
            atom.SetNumExplicitHs(1)


def get_data(lmdb_path, name=None):

    # lmdb_path='ligands/train.lmdb'

    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )
    txn = env.begin()
    keys = list(txn.cursor().iternext(values=False))

    smiles_data = []
    for idx in keys:
        datapoint_pickled = txn.get(idx)
        data = pickle.loads(datapoint_pickled)
        smiles_data.append({"smiles": data["smi"], "target": data["target"]})

    if name in ["clintox", "tox21", "toxcast", "sider", "pcba", "muv"]:
        for i in range(len(smiles_data)):
            smiles_data[i]["target"] = [list(smiles_data[i]["target"])]

    return smiles_data


def collect_and_save(path, fold):

    f = os.listdir(path)
    t = [i for i in f if fold + "_p" in i and i.endswith("pkl")]

    data = []
    for i in t:
        data.extend(pd.read_pickle(path + "/" + i))

    with open(f"{path}/{fold}.pkl", "wb") as f:
        pickle.dump(data, f)
    # return data


def save_ds_parts(data_creater=None, ds=None, output_dir=None, name=None, fold=None):

    if isinstance(ds, pd.DataFrame):
        ds.reset_index(drop=True, inplace=True)

        n = len(ds) // 1000
        parts = np.array_split(ds, n)

        for ipart, part in enumerate(parts):
            # ds_tmp = [ds[i] for i in ids]

            ds_tmp = data_creater.get_ft_dataset(part)
            ds_tmp = extract_data(ds_tmp)
            save_path = f"{output_dir}/{name}/{fold}_p_{ipart}"
            save_datasets(ds_tmp, save_path)
        if name:
            collect_and_save(f"{output_dir}/{name}", fold)
        else:
            collect_and_save(f"{output_dir}", fold)

    else:

        n = len(ds) // 1000
        id_list = np.array_split(np.arange(len(ds)), n)

        for ipart, ids in enumerate(id_list):
            ds_tmp = [ds[i] for i in ids]

            ds_tmp = data_creater.get_ft_dataset(ds_tmp)
            ds_tmp = extract_data(ds_tmp)
            save_path = f"{output_dir}/{name}/{fold}_p_{ipart}"
            save_datasets(ds_tmp, save_path)
        if name:
            collect_and_save(f"{output_dir}/{name}", fold)
        else:
            collect_and_save(f"{output_dir}", fold)
