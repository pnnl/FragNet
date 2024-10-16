import torch
import torch.nn as nn
from fragnet.train.utils import EarlyStopping
import torch
from torch.utils.data import DataLoader
from fragnet.dataset.dataset import  load_data_parts
import argparse
import numpy as np
from omegaconf import OmegaConf
import pickle
from torch.utils.tensorboard import SummaryWriter
from fragnet.model.gat.pretrain_heads import FragNetPreTrain, FragNetPreTrainMasked, FragNetPreTrainMasked2
from fragnet.train.pretrain.pretrain_utils import Trainer
from sklearn.model_selection import train_test_split
from fragnet.dataset.data import collate_fn_pt as collate_fn
import pandas as pd

def seed_everything(seed: int):
    import random, os
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    

def save_ds_smiles(ds,name, exp_dir):
    smiles = [i.smiles for i in ds]
    with open(exp_dir+f'/{name}_smiles.pkl', 'wb') as f:
        pickle.dump(smiles, f)

def load_train_val_dataset(args, ds):

    train_smiles = pd.read_pickle(args.pretrain.train_smiles)


    train_dataset, val_dataset = [], []
    for data in ds:
        if data.smiles in train_smiles:
            train_dataset.append(data)
        else:
            val_dataset.append(data)


    return train_dataset, val_dataset

def remove_duplicates_and_add(ds, path):
    t = load_data_parts(path)
    if len(ds) != 0:
        curr_smiles = [i.smiles for i in ds]
        new_ds = [i for i in t if i.smiles not in curr_smiles]
    else:
        new_ds=t
    ds += new_ds
    return ds

def save_predictions(trainer, loader, model, exp_dir, device, save_name='test_res', loss_type='mse'):
    score, true, pred = trainer.test(model=model, loader=loader, device=device)
    smiles = [i.smiles for i in loader.dataset]

    if loss_type=='mse':
        print(f'{save_name} rmse: ', score**.5)
        res = {'acc': score**.5, 'true': true, 'pred': pred, 'smiles': smiles}
    elif loss_type=='cel':
        # score = roc_auc_score(true, pred[:,1])
        print(f'{save_name} auc: ', score)
        res = {'acc': score, 'true': true, 'pred': pred, 'smiles': smiles}

    with open(f"{exp_dir}/{save_name}.pkl", 'wb') as f:
        pickle.dump(res,f )




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="configuration file *.yml", type=str, required=False, default='config.yaml')
    args = parser.parse_args()

    if args.config:
        opt = OmegaConf.load(args.config)
        OmegaConf.resolve(opt)
        args=opt

    seed_everything(args.seed)
    exp_dir = args['exp_dir']
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    writer = SummaryWriter(exp_dir+'/runs')

    pt_chkpoint_name = args.pretrain.chkpoint_name #'exps/pt_rings'

    if args.model_version == 'gat2':
        model = FragNetPreTrain(num_layer=args.pretrain.num_layer, 
                                drop_ratio=args.pretrain.drop_ratio,
                                    num_heads=args.pretrain.num_heads, 
                                    emb_dim=args.pretrain.emb_dim,
                                    atom_features=args.atom_features, 
                                    frag_features=args.frag_features, 
                                    edge_features=args.edge_features,
                                fedge_in=args.fedge_in, 
                                fbond_edge_in=args.fbond_edge_in
                               )
    elif args.model_version == 'gat2_masked':
        model = FragNetPreTrainMasked(num_layer=args.pretrain.num_layer, 
                        drop_ratio=args.pretrain.drop_ratio,
                            num_heads=args.pretrain.num_heads, 
                            emb_dim=args.pretrain.emb_dim,
                            atom_features=args.atom_features, 
                            frag_features=args.frag_features, 
                            edge_features=args.edge_features,
                            fedge_in=args.fedge_in, 
                            fbond_edge_in=args.fbond_edge_in)
        
    elif args.model_version == 'gat2_masked2':
        model = FragNetPreTrainMasked2(num_layer=args.pretrain.num_layer, 
                        drop_ratio=args.pretrain.drop_ratio,
                            num_heads=args.pretrain.num_heads, 
                            emb_dim=args.pretrain.emb_dim,
                            atom_features=args.atom_features, 
                            frag_features=args.frag_features, 
                            edge_features=args.edge_features,
                            fedge_in=args.fedge_in, 
                            fbond_edge_in=args.fbond_edge_in)
        
    if args.pretrain.saved_checkpoint:
        model.load_state_dict(torch.load(args.pretrain.saved_checkpoint))
        
    ds=[]
    for path in args.pretrain.data:
        ds = remove_duplicates_and_add(ds, path)


    if args.pretrain.train_smiles:
        train_dataset, val_dataset = load_train_val_dataset(args, ds)
    else:
        train_dataset, val_dataset = train_test_split(ds, test_size=.1, random_state=42)

    # NOTE: Start new runs in a new directory    
    save_ds_smiles(train_dataset, 'train', args.exp_dir)
    save_ds_smiles(val_dataset, 'val', args.exp_dir)

    print('number of data points: ', len(ds))
    writer.add_scalar('number of data points: ', len(ds))
    
    train_loader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=args.pretrain.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=args.pretrain.batch_size, shuffle=False, drop_last=False)
    

    model.to(device);
    early_stopping = EarlyStopping(patience=args.pretrain.es_patience, verbose=True, chkpoint_name=pt_chkpoint_name)


    if args.pretrain.loss == 'mse':
        loss_fn = nn.MSELoss()
    elif args.pretrain.loss == 'cel':
        loss_fn = nn.CrossEntropyLoss()

    trainer = Trainer(loss_fn=loss_fn)
        
    optimizer = torch.optim.Adam(model.parameters(), lr = args.pretrain.lr ) # before 1e-4

    for epoch in range(args.pretrain.n_epochs):

        train_loss = trainer.train(model=model, loader=train_loader, optimizer=optimizer, device=device)
        
        writer.add_scalar('Loss/train', train_loss, epoch)


        if epoch%5==0:
            val_loss = trainer.validate(model=model, loader=val_loader, device=device)
            print(train_loss, val_loss)
            writer.add_scalar('Loss/val', val_loss, epoch)

            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break  
