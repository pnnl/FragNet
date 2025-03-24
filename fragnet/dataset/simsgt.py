import torch
from torch_geometric.datasets import MoleculeNet
from .utils import save_datasets
import pandas as pd
from .dataset import FinetuneData
from .utils import extract_data


def create_moleculenet_dataset_simsgt(name, args):
    from splitters_simsgt import scaffold_split
    from loader_simsgt import MoleculeDataset

    d = MoleculeNet(f"{args.output_dir}/simsgt", name=name)
    dataset = MoleculeDataset(f"{args.output_dir}/simsgt/" + name, dataset=name)

    smiles_list = pd.read_csv(
        f"{args.output_dir}/simsgt/" + name + "/processed/smiles.csv", header=None
    )[0].tolist()
    (
        train_dataset,
        valid_dataset,
        test_dataset,
        (train_smiles, valid_smiles, test_smiles),
    ) = scaffold_split(
        dataset,
        smiles_list,
        null_value=0,
        frac_train=0.8,
        frac_valid=0.1,
        frac_test=0.1,
        return_smiles=True,
    )

    torch.save(train_dataset, f"{args.output_dir}/simsgt/{name}/train.pt")
    torch.save(valid_dataset, f"{args.output_dir}/simsgt/{name}/val.pt")
    torch.save(test_dataset, f"{args.output_dir}/simsgt/{name}/test.pt")

    dataset = FinetuneData(args.target_name)

    if not args.save_parts:

        ds = dataset.get_ft_dataset(train_dataset)
        ds = extract_data(ds)
        save_path = f"{args.output_dir}/simsgt/{name}/train"
        save_datasets(ds, save_path)

        ds = dataset.get_ft_dataset(valid_dataset)
        ds = extract_data(ds)
        save_path = f"{args.output_dir}/simsgt/{name}/val"
        save_datasets(ds, save_path)

        ds = dataset.get_ft_dataset(test_dataset)
        ds = extract_data(ds)
        save_path = f"{args.output_dir}/simsgt/{name}/test"
        save_datasets(ds, save_path)
