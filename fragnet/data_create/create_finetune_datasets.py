import argparse

from fragnet.dataset.moleculenet import create_moleculenet_dataset
from fragnet.dataset.general import create_general_dataset

# from fragnet.dataset.unimol import create_moleculenet_dataset_from_unimol_data
from fragnet.dataset.simsgt import create_moleculenet_dataset_simsgt
from fragnet.dataset.dta import create_dta_dataset
from fragnet.dataset.cdrp import create_cdrp_dataset
from fragnet.dataset.scaffold_split_from_df import create_scaffold_split_data_from_df

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        help="moleculenet, moleculenet-custom",
        type=str,
        required=False,
        default="moleculenet",
    )
    parser.add_argument(
        "--dataset_subset",
        help="esol, freesolv",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--output_dir", help="", type=str, required=False, default="finetune_data"
    )
    parser.add_argument(
        "--data_dir", help="", type=str, required=False, default="finetune_data"
    )
    parser.add_argument(
        "--frag_type", help="", type=str, required=False, default="brics"
    )
    parser.add_argument(
        "--use_molebert", help="", type=bool, required=False, default=False
    )
    parser.add_argument("--train_path", help="", type=str, required=False, default=None)
    parser.add_argument("--save_parts", help="", type=int, required=False, default=0)
    parser.add_argument("--val_path", help="", type=str, required=False, default=None)
    parser.add_argument("--test_path", help="", type=str, required=False, default=None)
    parser.add_argument("--target_name", help="", type=str, required=False, default="y")
    parser.add_argument("--data_type", help="", type=str, required=False, default="exp")
    parser.add_argument(
        "--multi_conf_data",
        help="create multiple conformers for a smiles",
        type=int,
        required=False,
        default=0,
    )
    parser.add_argument(
        "--use_genes",
        help="whether a gene subset is used for cdrp data",
        type=int,
        required=False,
        default=0,
    )

    args = parser.parse_args()

    print("args.use_genes: ", args.use_genes)
    print("args.multi_conf_data: ", args.multi_conf_data)

    if "gen" in args.dataset_name:

        create_general_dataset(args)
    elif args.dataset_name == "moleculenet":

        create_moleculenet_dataset("MoleculeNet", args.dataset_subset.lower(), args)
    elif args.dataset_name == "moleculedataset":
        create_moleculenet_dataset("MoleculeDataset", args.dataset_subset.lower(), args)

    elif args.dataset_name == "unimol":
        create_moleculenet_dataset_from_unimol_data(args)

    elif args.dataset_name == "simsgt":
        create_moleculenet_dataset_simsgt(args.dataset_subset.lower(), args)

    elif args.dataset_name in ["davis", "kiba"]:
        create_dta_dataset(args)

    elif args.dataset_name in ["cep", "malaria"]:
        create_scaffold_split_data_from_df(args)

    elif args.dataset_name in ["gdsc", "gdsc_full", "ccle"]:
        print("args.use_genes0: ", args.use_genes)
        create_cdrp_dataset(args)
