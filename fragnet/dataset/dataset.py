import torch
import time
from multiprocessing.pool import ThreadPool as Pool
from tqdm import tqdm
import pickle
import pandas as pd
import os
from ast import literal_eval
from rdkit import Chem
import torch_geometric
from torch_geometric.datasets import MoleculeNet
from pathlib import Path
from .fragments import get_3Dcoords2, get_3Dcoords
from .data import CreateData, CreateDataDTA
from .data import CreateDataCDRP


# pretrain dataset
def get_pt_dataset(df, data_type, maxiters=200, frag_type="brics"):

    """
    Creates a pretrain dataset.

    Args:
        df: DataFrame containing the SMILES
        data_type: determines the feature type
    """

    feature_type = "one_hot"
    create_bond_graph_data = True
    add_dhangles = True

    create_data = CreateData(
        data_type=data_type,
        create_bond_graph_data=create_bond_graph_data,
        add_dhangles=add_dhangles,
    )

    st = time.time()
    x = []
    n_data = len(df)
    if isinstance(df, pd.DataFrame):
        df.reset_index(drop=True, inplace=True)
    for i in tqdm(range(n_data)):
        smiles = df.loc[i, "smiles"]

        res = get_3Dcoords2(smiles, maxiters=maxiters)
        if res != None:
            mol, conf_res = res

            for j in range(len(conf_res)):
                E = conf_res[j][1]
                conf = mol.GetConformer(id=j)
                x.append([smiles, [E], mol, conf, frag_type])

    print(time.time() - st)

    st = time.time()
    train_dataset = Pool().imap(create_data.create_data_point, x)
    print(time.time() - st)

    return train_dataset


class FinetuneData:

    """
    The main class to create finetune data

    Attributes:
        frag_type (str): fragment type (brics or murcko).
        target_name (str): name of the target proprty.
        data_type (str): determines the features.
    """
    def __init__(self, target_name, data_type, **kwargs):
        self.frag_type = kwargs.get("frag_type", "brics")
        self.target = target_name
        self.data_type = data_type
        self.create_data = CreateData(
            data_type=data_type, create_bond_graph_data=True, add_dhangles=False
        )

    def get_ft_dataset(self, df):

        st = time.time()
        x = []
        n_data = len(df)
        for i in tqdm(range(n_data)):

            if isinstance(df, pd.DataFrame):
                smiles = df.loc[i, "smiles"]
                y = [df.loc[i, self.target]]
            elif isinstance(df, torch_geometric.datasets.molecule_net.MoleculeNet):
                smiles = df[i].smiles
                y = list(df[i][self.target])
            elif isinstance(df, list):
                smiles = df[i]["smiles"]
                y = list(df[i][self.target])

            mol = get_3Dcoords(smiles)
            conf = mol.GetConformer(id=0)

            x.append([smiles, y, mol, conf, self.frag_type])

        print(time.time() - st)

        st = time.time()
        train_dataset = Pool().imap(self.create_data.create_data_point, x)
        print(time.time() - st)

        return train_dataset


class FinetuneDataDTA:

    def __init__(self, target_name, data_type):
        self.target = target_name
        self.data_type = data_type
        self.create_data = CreateDataDTA(
            data_type=data_type, create_bond_graph_data=True, add_dhangles=False
        )

    def get_ft_dataset(self, df):

        if isinstance(df, pd.DataFrame):
            df.reset_index(drop=True, inplace=True)

        st = time.time()
        x = []
        n_data = len(df)
        for i in tqdm(range(n_data)):

            if isinstance(df, pd.DataFrame):

                smiles = df.loc[i, "smiles"]
                y = [df.loc[i, self.target]]
                protein = df.loc[i, "protein"]

            mol = get_3Dcoords(smiles)
            conf = mol.GetConformer(id=0)

            x.append([smiles, y, mol, conf, protein])

        print(time.time() - st)

        st = time.time()
        train_dataset = Pool().imap(self.create_data.create_data_point, x)
        print(time.time() - st)

        return train_dataset


class FinetuneDataCDRP:

    """
    To create finetune data for Cancer Drug Response Prediction
    """
    def __init__(self, target_name, data_type, args, use_genes=False):
        self.target = target_name
        self.data_type = data_type
        self.data_dir = args.data_dir

        self.use_genes = use_genes
        if use_genes is True:
            self.genes = self.load_ext_genes()
        self.gene_expr_data = self.load_gene_expr_data()

        self.create_data = CreateDataCDRP(
            data_type=data_type, create_bond_graph_data=True, add_dhangles=False
        )

    def load_ext_genes(self):
        genes = pd.read_csv(os.path.join(self.data_dir, "landmark_genes"), header=None)
        genes = genes.values.ravel().tolist()
        return genes

    def load_gene_expr_data(self):
        rnafile = self.data_dir + "/Cell_line_RMA_proc_basalExp.txt"
        rnadata = pd.read_csv(rnafile, sep="\t")

        if self.use_genes is True:
            gene_expr = rnadata[rnadata.GENE_SYMBOLS.isin(self.genes)]
        else:
            gene_expr = rnadata

        cell_line_cols = [i for i in gene_expr.columns if i.startswith("DATA")]
        gene_expr = gene_expr.loc[:, cell_line_cols]
        gene_expr = gene_expr.T

        return gene_expr

    def get_ft_dataset(self, df):

        if isinstance(df, pd.DataFrame):
            df.reset_index(drop=True, inplace=True)

        st = time.time()
        x = []
        n_data = len(df)
        for i in tqdm(range(n_data)):

            if isinstance(df, pd.DataFrame):

                smiles = df.loc[i, "smiles"]
                y = [df.loc[i, self.target]]
                cosmic_id = df.loc[i, "COSMIC_ID"]
                cosmic_id = "DATA." + str(cosmic_id)

                gene_expr = self.gene_expr_data.loc[cosmic_id, :].values.tolist()

            mol = get_3Dcoords(smiles)
            conf = mol.GetConformer(id=0)

            x.append([smiles, y, mol, conf, gene_expr])

        print(time.time() - st)

        st = time.time()
        train_dataset = Pool().imap(self.create_data.create_data_point, x)
        print(time.time() - st)

        return train_dataset


class FinetuneMultiConfData:

    """
    To create finetune data containing multiple conformers
    """
    def __init__(self, target_name, data_type, **kwargs):
        self.frag_type = kwargs["frag_type"]
        self.maxiters = 500
        self.target = target_name
        self.data_type = data_type
        self.create_data = CreateData(
            data_type=data_type, create_bond_graph_data=True, add_dhangles=False
        )

    def get_ft_dataset(self, df):

        st = time.time()
        x = []
        n_data = len(df)
        for i in tqdm(range(n_data)):

            if isinstance(df, pd.DataFrame):
                smiles = df.loc[i, "smiles"]
                y = [df.loc[i, self.target]]
            elif isinstance(df, torch_geometric.datasets.molecule_net.MoleculeNet):
                smiles = df[i].smiles
                y = list(df[i][self.target])
            elif isinstance(df, list):
                smiles = df[i]["smiles"]
                y = list(df[i][self.target])

            res = get_3Dcoords2(smiles, numconf=10, maxiters=self.maxiters)
            if res != None:
                mol, conf_res = res

                for j in range(len(conf_res)):
                    conf = mol.GetConformer(id=j)
                    x.append([smiles, y, mol, conf, self.frag_type])

        print(time.time() - st)

        st = time.time()
        train_dataset = Pool().imap(self.create_data.create_data_point, x)
        print(time.time() - st)

        return train_dataset


def load_pickle_dataset(path):
    with open(f"{path}", "rb") as f:
        dataset = pickle.load(f)
    dataset = [d for d in dataset if d]
    return dataset


def load_data_parts(path, select_name=None):
    parts = os.listdir(path)

    if select_name:
        parts = [p for p in parts if select_name in p]

    print(parts)
    datasets = []
    for i in parts:
        d = load_pickle_dataset(Path(path) / Path(i))
        datasets += d

    return datasets


def get_raw_data(dataset):

    smiles = [i.smiles for i in dataset]
    y = [i.y.tolist() for i in dataset]

    return pd.DataFrame(zip(smiles, y), columns=["smiles", "target"])


class LoadDataSets:

    """
    Class for dataset loading. 
    Used in train/pretrain/pretrain_gat_mol.py 
    """
    def __init__(self):
        self.remove_duplicates_and_add = self.remove_duplicates_and_add1

    def load_datasets(self, opt):
        train_dataset, val_dataset, test_dataset = [], [], []
        for k, v in opt.pretrain.datasets.items():
            if "train" in v:
                path = v.train.path
                name = v.train.name
                include = None
                if "include" in v.train:
                    include = literal_eval(v.train.include)
                    include = range(include[0], include[1])
                train_dataset = self.remove_duplicates_and_add(
                    train_dataset, path, name, include=include
                )
            if "val" in v:
                path = v.val.path
                name = v.val.name
                include = None
                if "include" in v.val:
                    include = literal_eval(v.val.include)
                    include = range(include[0], include[1])
                val_dataset = self.remove_duplicates_and_add(
                    val_dataset, path, name, include=include
                )
            if "test" in v:
                path = v.test.path
                name = v.test.name
                include = None
                if "include" in v.test:
                    include = literal_eval(v.test.include)
                    include = range(include[0], include[1])
                test_dataset = self.remove_duplicates_and_add(
                    test_dataset, path, name, include=include
                )

        return train_dataset, val_dataset, test_dataset

    def remove_duplicates_and_add1(self, ds, path, name, include):
        t = load_data_parts(path, name, include=include)
        if len(ds) != []:
            curr_smiles = [i.smiles for i in ds]
            new_ds = [i for i in t if i.smiles not in curr_smiles]
        else:
            new_ds = t
        ds += new_ds
        return ds

    def remove_duplicates_and_add_ext_prop(self, ds, path, name, include):
        t = load_data_parts(path, name, include=include)
        if len(ds) != []:
            curr_smiles = [i.smiles for i in ds]
            new_ds = [i for i in t if i.smiles not in curr_smiles]
        else:
            new_ds = t

        Pool().imap(self.add_prop, new_ds)
        ds += new_ds
        return ds

    def add_prop(self, datapoint):

        smiles = datapoint.smiles
        y = self.props.loc[smiles, self.propv].values.tolist()
        datapoint.y = torch.tensor(y, dtype=torch.float)
