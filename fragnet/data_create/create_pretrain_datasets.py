import pandas as pd
from fragnet.dataset.dataset import get_pt_dataset
from fragnet.dataset.utils import extract_data, save_datasets
import argparse
import os
import logging
import logging.config

def continuous_creation(args):

    l_limit = args.low
    h_limit = args.high
    n_rows =  args.n_rows

    for i in range(l_limit, h_limit, n_rows):
        start_id = i
        end_id = i+n_rows-1
        dfi = df.loc[start_id : end_id]


        ds = get_pt_dataset(dfi, args.data_type, maxiters=args.maxiters)
        ds = extract_data(ds)
        save_path = f'{args.save_path}/pt_{start_id}_{end_id}'
        save_datasets(ds, save_path)

        print(dfi.shape)


def create_from_ids(args):

    curr = pd.read_pickle('pretrain_data/unimol_exp1s/curr_tmp.pkl')
    full = pd.read_pickle('../fragnet1.2/pretrain_data/unimol/file_list.pkl')
    new = set(full).difference(curr)
    new = list(new)
    new = new[args.low: args.high]
    start_ids = [ int(i.split('_')[1]) for i in new]

    for start_id in start_ids:
        end_id = start_id+n_rows-1
        dfi = df.loc[start_id : end_id]

        ds = get_pt_dataset(dfi, args.data_type)
        ds = extract_data(ds)
        save_path = f'{args.save_path}/pt_{start_id}_{end_id}'
        logger.info(save_path)
        save_datasets(ds, save_path)

        print(dfi.shape)



def continuous_creation_add_new(args):

    l_limit = args.low
    h_limit = args.high
    n_rows =  args.n_rows

    for i in range(l_limit, h_limit, n_rows):
        start_id = i
        end_id = i+n_rows-1
        dfi = df.loc[start_id : end_id]

        save_path = f'{args.save_path}/pt_{start_id}_{end_id}'

        curr_data = pd.read_pickle( save_path +  '.pkl' )
        curr_smiles = [d.smiles for d in curr_data]

        dfi_rem = dfi[~dfi.smiles.isin(curr_smiles)]

        ds = get_pt_dataset(dfi_rem, args.data_type, maxiters=args.maxiters, frag_type=args.frag_type)
        ds = extract_data(ds)

        ds_updated = ds+curr_data
        save_datasets(ds_updated, save_path)
        print(dfi.shape)
        


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', help="", type=str, required=False, default="pretrain_data/unimol/ds")
    parser.add_argument('--low', help="", type=int, required=False, default=None)
    parser.add_argument('--high', help="", type=int, required=False, default=None)
    parser.add_argument('--data_type', help="", type=str, required=False, default='exp')
    parser.add_argument('--raw_data_path', help="", type=str, required=False, default=None)
    parser.add_argument('--calc_type', help="", type=str, required=False, default='scratch')
    parser.add_argument('--maxiters', help="", type=int, required=False, default=200)
    parser.add_argument('--frag_type', help="", type=str, required=False, default='brics')

    
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

    if args.raw_data_path == None:
        df = pd.read_csv('pretrain_data/input/train_no_modulus.csv')
    else:
        df = pd.read_csv(args.raw_data_path)

    n_rows = 1000
    if not args.low:
        args.low = 0
    if not args.high:
        args.high = len(df)
    args.n_rows = n_rows

    if args.calc_type == 'scratch':
        continuous_creation(args)
    elif args.calc_type == 'add':
        continuous_creation_add_new(args)







