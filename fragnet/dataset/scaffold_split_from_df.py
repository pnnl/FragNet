import os
import pandas as pd
from .dataset import FinetuneData
from .utils import extract_data
from .utils import save_datasets


def create_scaffold_split_data_from_df(args):

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    ds = pd.read_csv(args.data_dir)
    ds.reset_index(drop=True, inplace=True)

    from splitters_molebert import scaffold_split

    smiles = ds.smiles.values.tolist()

    train, val, test, (train_smiles, valid_smiles, test_smiles) = scaffold_split(
        ds, smiles, return_smiles=True
    )

    train.reset_index(drop=True, inplace=True)
    val.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    dataset = FinetuneData(target_name=args.target_name, data_type=args.data_type)

    if not args.save_parts:

        train.to_csv(f"{args.output_dir}/train.csv", index=False)
        ds = dataset.get_ft_dataset(train)
        ds = extract_data(ds)
        save_path = f"{args.output_dir}/train"
        save_datasets(ds, save_path)

        val.to_csv(f"{args.output_dir}/val.csv", index=False)
        ds = dataset.get_ft_dataset(val)
        ds = extract_data(ds)
        save_path = f"{args.output_dir}/val"
        save_datasets(ds, save_path)

        test.to_csv(f"{args.output_dir}/test.csv", index=False)
        ds = dataset.get_ft_dataset(test)
        ds = extract_data(ds)
        save_path = f"{args.output_dir}/test"
        save_datasets(ds, save_path)
