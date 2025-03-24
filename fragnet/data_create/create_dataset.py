import pandas as pd
from dataset import save_dataset_parts, save_dataset
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        help="dataframe or csv file",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--train_path",
        help="path to train input data",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--val_path",
        help="path to val input data",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--test_path",
        help="path to test input data",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument("--start_row", help="", type=int, required=False, default=None)
    parser.add_argument("--end_row", help="", type=int, required=False, default=None)
    parser.add_argument("--start_id", help="", type=int, required=False, default=0)
    parser.add_argument(
        "--rows_per_part", help="", type=int, required=False, default=1000
    )
    parser.add_argument(
        "--create_bond_graph_data", help="", type=bool, required=False, default=True
    )
    parser.add_argument(
        "--add_dhangles", help="", type=bool, required=False, default=False
    )
    parser.add_argument(
        "--feature_type",
        help="one_hot or embed",
        type=str,
        required=False,
        default="one_hot",
    )
    parser.add_argument(
        "--target",
        help="target property",
        type=str,
        required=False,
        nargs="+",
        default="log_sol",
    )
    parser.add_argument(
        "--save_path",
        help="folder where the data is saved",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--save_name", help="saving file name", type=str, required=False, default=None
    )
    args = parser.parse_args()

    dataset_args = {
        "create_bond_graph_data": args.create_bond_graph_data,
        "add_dhangles": args.add_dhangles,
        "feature_type": args.feature_type,
        "target": args.target,
        "save_path": args.save_path,
        "save_name": args.save_name,
        "start_id": args.start_id,
    }

    if args.train_path:
        train = pd.read_csv(args.train_path)
        train.to_csv(f"{args.save_path}/train.csv", index=False)
        save_dataset(
            df=train,
            save_path=f"{args.save_path}",
            save_name=f"train",
            target=args.target,
            feature_type=args.feature_type,
            create_bond_graph_data=args.create_bond_graph_data,
        )

    elif args.val_path:
        val = pd.read_csv(args.val_path)
        val.to_csv(f"{args.save_path}/val.csv", index=False)
        save_dataset(
            df=val,
            save_path=f"{args.save_path}",
            save_name=f"val",
            target=args.target,
            feature_type=args.feature_type,
            create_bond_graph_data=args.create_bond_graph_data,
        )

    elif args.test_path:
        test = pd.read_csv(args.test_path)
        test.to_csv(f"{args.save_path}/test.csv", index=False)
        save_dataset(
            df=test,
            save_path=f"{args.save_path}",
            save_name=f"test",
            target=args.target,
            feature_type=args.feature_type,
            create_bond_graph_data=args.create_bond_graph_data,
        )

    else:
        save_dataset_parts(
            args.data_path,
            start_row=args.start_row,
            end_row=args.end_row,
            rows_per_part=args.rows_per_part,
            dataset_args=dataset_args,
        )
