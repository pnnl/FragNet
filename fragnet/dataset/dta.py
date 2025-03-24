import os
import pandas as pd
from .dataset import FinetuneDataDTA
from .utils import extract_data, save_datasets, save_ds_parts


def create_dta_dataset(args):

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    dataset = FinetuneDataDTA(target_name=args.target_name, data_type=args.data_type)

    if args.save_parts == 0:

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

    else:
        save_ds_parts(
            data_creater=dataset, ds=train, output_dir=args.output_dir, fold="train"
        )
        save_ds_parts(
            data_creater=dataset, ds=val, output_dir=args.output_dir, fold="val"
        )
        save_ds_parts(
            data_creater=dataset, ds=test, output_dir=args.output_dir, fold="test"
        )
