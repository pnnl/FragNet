import os
import torch
from torch_geometric.datasets import MoleculeNet

from .utils import save_datasets
from .dataset import FinetuneData, FinetuneMultiConfData
from .utils import extract_data
from .splitters import ScaffoldSplitter
from .custom_dataset import MoleculeDataset
from .utils import save_ds_parts
import pandas as pd


def create_general_dataset(args):

    if args.multi_conf_data == 1:
        print("generating multi-conf data")
        dataset = FinetuneMultiConfData(
            target_name=args.target_name,
            data_type=args.data_type,
            frag_type=args.frag_type,
        )
    else:
        print("generating single-conf data")
        dataset = FinetuneData(
            target_name=args.target_name,
            data_type=args.data_type,
            frag_type=args.frag_type,
        )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    if args.train_path:
        train = pd.read_csv(args.train_path)
        train.to_csv(f"{args.output_dir}/train.csv", index=False)
        ds = dataset.get_ft_dataset(train)
        ds = extract_data(ds)
        save_path = f"{args.output_dir}/train"
        save_datasets(ds, save_path)

    if args.val_path:
        val = pd.read_csv(args.val_path)
        val.to_csv(f"{args.output_dir}/val.csv", index=False)
        ds = dataset.get_ft_dataset(val)
        ds = extract_data(ds)
        save_path = f"{args.output_dir}/val"
        save_datasets(ds, save_path)

    if args.test_path:
        test = pd.read_csv(args.test_path)
        test.to_csv(f"{args.output_dir}/test.csv", index=False)
        ds = dataset.get_ft_dataset(test)
        ds = extract_data(ds)
        save_path = f"{args.output_dir}/test"
        save_datasets(ds, save_path)
