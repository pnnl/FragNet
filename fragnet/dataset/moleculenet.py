
import os
import torch
from torch_geometric.datasets import MoleculeNet

from .utils import save_datasets
from .dataset import FinetuneData, FinetuneMultiConfData
from .utils import extract_data
from .splitters import ScaffoldSplitter
from .custom_dataset import MoleculeDataset
from .utils import save_ds_parts


def create_moleculenet_dataset(dstype, name, args):
    

    if not os.path.exists(args.output_dir+'/'+name):
        os.makedirs(args.output_dir+'/'+name, exist_ok=True)

    if dstype=='MoleculeNet':
        ds = MoleculeNet(f'{args.data_dir}',  name=name)
    elif dstype=='MoleculeDataset':
        _ = MoleculeNet(f'{args.data_dir}',  name=name)
        dataset = MoleculeDataset(name, args.data_dir)
        ds = dataset.get_data()
        
    if not args.use_molebert:
        scaffold_split = ScaffoldSplitter()
        train, val, test = scaffold_split.split(dataset=dataset, include_chirality=True)
    elif args.use_molebert:
        from .splitters_molebert import scaffold_split
        smiles = [i.smiles for i in ds]
        train, val, test, (train_smiles, valid_smiles, test_smiles) = scaffold_split(ds, smiles, return_smiles=True)


    torch.save(train, f'{args.output_dir}/{name}/train.pt')
    torch.save(val, f'{args.output_dir}/{name}/val.pt')
    torch.save(test, f'{args.output_dir}/{name}/test.pt')

    if args.multi_conf_data:
        dataset = FinetuneMultiConfData(args.target_name, args.data_type)
    else:
        dataset = FinetuneData(args.target_name, args.data_type, frag_type=args.frag_type ) 
    
    if not args.save_parts:

        ds = dataset.get_ft_dataset(train)
        ds = extract_data(ds)
        save_path = f'{args.output_dir}/{name}/train'
        save_datasets(ds, save_path)


        ds = dataset.get_ft_dataset(val)
        ds = extract_data(ds)
        save_path = f'{args.output_dir}/{name}/val'
        save_datasets(ds, save_path)
        
        ds = dataset.get_ft_dataset(test)
        ds = extract_data(ds)
        save_path = f'{args.output_dir}/{name}/test'
        save_datasets(ds, save_path)


    elif args.save_parts:
        save_ds_parts(data_creater=dataset, ds=train, output_dir=args.output_dir, name=name, fold='train')
        save_ds_parts(data_creater=dataset, ds=val, output_dir=args.output_dir, name=name, fold='val')
        save_ds_parts(data_creater=dataset, ds=test, output_dir=args.output_dir, name=name, fold='test')
