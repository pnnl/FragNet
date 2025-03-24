import pandas as pd
from torch_geometric.datasets import MoleculeNet
from .utils import remove_non_mols
from torch_geometric.data import Data


class MoleculeDataset:
    def __init__(self, name, data_dir):
        self.name = name
        self.data_dir = data_dir

    def get_data(self):
        if self.name == "tox21":
            return self.get_tox21()
        elif self.name == "toxcast":
            return self.get_toxcast()
        elif self.name == "clintox":
            return self.get_clintox()
        elif self.name == "sider":
            return self.get_sider()
        elif self.name == "bbbp":
            return self.get_bbbp()
        elif self.name == "hiv":
            return self.get_hiv_dataset()
        elif self.name == "muv":
            return self.get_muv()

    def get_muv(self):
        from loader_molebert import _load_muv_dataset

        raw_path = f"{self.data_dir}/muv/raw/muv.csv"
        smiles_list, rdkit_mol_objs, labels = _load_muv_dataset(raw_path)

        assert len(smiles_list) == len(labels)
        data = []
        for i in range(len(smiles_list)):
            y = [list(labels[i])]
            smiles = smiles_list[i]
            if smiles != None:
                data.append(Data(smiles=smiles, y=y))

        return data

    def get_tox21(self):
        from loader_molebert import _load_tox21_dataset

        raw_path = f"{self.data_dir}/tox21/raw/tox21.csv"
        smiles_list, rdkit_mol_objs, labels = _load_tox21_dataset(raw_path)

        assert len(smiles_list) == len(labels)
        data = []
        for i in range(len(smiles_list)):
            y = [list(labels[i])]
            smiles = smiles_list[i]
            if smiles != None:
                data.append(Data(smiles=smiles, y=y))

        return data

    def get_hiv_dataset(self):
        from loader_molebert import _load_hiv_dataset

        raw_path = f"{self.data_dir}/hiv/raw/HIV.csv"

        smiles_list, rdkit_mol_objs, labels = _load_hiv_dataset(raw_path)

        assert len(smiles_list) == len(labels)
        data = []
        for i in range(len(smiles_list)):

            y = [labels[i]]
            smiles = smiles_list[i]
            if smiles != None:
                data.append(Data(smiles=smiles, y=y))

        return data

    def get_toxcast(self):
        from loader_molebert import _load_toxcast_dataset

        raw_path = f"{self.data_dir}/toxcast/raw/toxcast_data.csv"

        smiles_list, rdkit_mol_objs, labels = _load_toxcast_dataset(raw_path)

        assert len(smiles_list) == len(labels)
        data = []
        for i in range(len(smiles_list)):
            y = [list(labels[i])]
            smiles = smiles_list[i]
            if smiles != None:
                data.append(Data(smiles=smiles, y=y))

        return data

    def get_clintox(self):
        from loader_molebert import _load_clintox_dataset

        raw_path = f"{self.data_dir}/clintox/raw/clintox.csv"

        smiles_list, rdkit_mol_objs, labels = _load_clintox_dataset(raw_path)

        assert len(smiles_list) == len(labels)
        data = []
        for i in range(len(smiles_list)):
            y = [list(labels[i])]
            smiles = smiles_list[i]
            if smiles != None:
                data.append(Data(smiles=smiles, y=y))

        return data

    def get_bbbp(self):
        from loader_molebert import _load_bbbp_dataset

        raw_path = f"{self.data_dir}/bbbp/raw/BBBP.csv"

        smiles_list, rdkit_mol_objs, labels = _load_bbbp_dataset(raw_path)

        assert len(smiles_list) == len(labels)
        data = []
        for i in range(len(smiles_list)):
            y = [labels[i]]
            smiles = smiles_list[i]
            if smiles != None:
                data.append(Data(smiles=smiles, y=y))

        return data

    def get_sider(self):
        from loader_molebert import _load_sider_dataset

        dataset = MoleculeNet(self.data_dir, name=self.name)
        raw_path = f"{self.data_dir}/sider/raw/sider.csv"
        smiles_list, rdkit_mol_objs, labels = _load_sider_dataset(raw_path)

        assert len(smiles_list) == len(labels)
        data = []
        for i in range(len(smiles_list)):
            y = [list(labels[i, :])]
            smiles = smiles_list[i]

            if smiles != None:
                data.append(Data(smiles=smiles, y=y))

        return data

    def get_pcba(self):

        dataset = MoleculeNet(self.data_dir, name=self.name)
        df = pd.read_csv(f"{self.data_dir}/sider/raw/pcba.csv")
        df = remove_non_mols(df)
        df = df.fillna(-1)
        df.reset_index(drop=True, inplace=True)

        data = []
        for i in df.index:
            y = [df.iloc[i, :128].values.tolist()]
            smiles = df.loc[i, "smiles"]
            data.append(Data(smiles=smiles, y=y))

        return data
