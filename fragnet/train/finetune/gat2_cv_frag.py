# from gat import FragNetFineTune
import torch
from fragnet.dataset.dataset import load_pickle_dataset
import torch.nn as nn
from fragnet.train.utils import EarlyStopping
import torch
from fragnet.dataset.data import collate_fn
from torch.utils.data import DataLoader
import pandas as pd
import argparse
from fragnet.train.utils import TrainerFineTune as Trainer
import numpy as np
from omegaconf import OmegaConf
import pickle
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.model_selection import KFold
import os
from sklearn.model_selection import train_test_split

def seed_everything(seed: int):
    import random, os
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(26)

def get_predictions(trainer, loader, model, device):
    mse, true, pred = trainer.test(model=model, loader=loader, device=device)
    smiles = [i.smiles for i in loader.dataset]    
    res = {'smiles': smiles,  'true': true, 'pred': pred}    
    return res

def save_predictions(exp_dir, save_name, res):
    
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
    os.makedirs(exp_dir, exist_ok=True)
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    
    pt_chkpoint_name = args.pretrain.chkpoint_name #'exps/pt_rings'

    if args.model_version == 'gat':
        from fragnet.model.gat.gat import FragNetFineTune
        print('loaded from gat')
    elif args.model_version=='gat2':
        from fragnet.model.gat.gat2 import FragNetFineTune
        print('loaded from gat2')
    
    
    model = FragNetFineTune(n_classes=args.finetune.model.n_classes, 
                            atom_features=args.atom_features, 
                            frag_features=args.frag_features, 
                            edge_features=args.edge_features,
                            num_layer=args.finetune.model.num_layer, 
                            drop_ratio=args.finetune.model.drop_ratio,
                                num_heads=args.finetune.model.num_heads, 
                                emb_dim=args.finetune.model.emb_dim,
                                h1=args.finetune.model.h1,
                                h2=args.finetune.model.h2,
                                h3=args.finetune.model.h3,
                                h4=args.finetune.model.h4,
                                act=args.finetune.model.act,
                                fthead=args.finetune.model.fthead
                                )
    trainer = Trainer(target_type=args.finetune.target_type)
    
    pt_chkpoint_name = args.pretrain.chkpoint_name #'exps/pt_rings'
    if pt_chkpoint_name:

        from fragnet.model.gat.gat2_pretrain import FragNetPreTrain
        modelpt = FragNetPreTrain(num_layer=args.finetune.model.num_layer, 
                                  drop_ratio=args.finetune.model.drop_ratio, 
                                  num_heads=args.finetune.model.num_heads, 
                                  emb_dim=args.finetune.model.emb_dim,
                 atom_features=args.atom_features, frag_features=args.frag_features, edge_features=args.edge_features)
        

        modelpt.load_state_dict(torch.load(pt_chkpoint_name, map_location=torch.device(device)))

        print('loading pretrained weights')
        model.pretrain.load_state_dict(modelpt.pretrain.state_dict())
        print('weights loaded')
    else:
        print('no pretrained weights')


    train_dataset2 = load_pickle_dataset(args.finetune.train.path)

    kf = KFold(n_splits=5)
    train_val = train_dataset2
    
    for icv, (train_index, test_index) in enumerate(kf.split(train_val)):
        train_index, val_index = train_test_split(train_index, test_size=.1)

        train_ds = [train_val[i] for i in train_index] 
        val_ds = [train_val[i] for i in val_index] 
        test_ds = [train_val[i] for i in test_index] 
        
        train_loader = DataLoader(train_ds, collate_fn=collate_fn, batch_size=args.finetune.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, collate_fn=collate_fn, batch_size=args.finetune.batch_size, shuffle=False, drop_last=False)
        test_loader = DataLoader(test_ds, collate_fn=collate_fn, batch_size=args.finetune.batch_size, shuffle=False, drop_last=False)


        trainer = Trainer(target_type=args.finetune.target_type)


        model.to(device);
        ft_chk_point = os.path.join(args.exp_dir, f'cv_{icv}.pt')
        early_stopping = EarlyStopping(patience=args.finetune.es_patience, verbose=True, chkpoint_name=ft_chk_point)


        if args.finetune.loss == 'mse':
            loss_fn = nn.MSELoss()
        elif args.finetune.loss == 'cel':
            loss_fn = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr = args.finetune.lr ) # before 1e-4
        if args.finetune.use_schedular:
            scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        else:
            scheduler=None

        for epoch in range(args.finetune.n_epochs):

            train_loss = trainer.train(model=model, loader=train_loader, optimizer=optimizer, 
                                       scheduler=scheduler, device=device, val_loader=val_loader)
            val_loss, _, _ = trainer.test(model=model, loader=val_loader, device=device)


            print(train_loss, val_loss)


            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break


        model.load_state_dict(torch.load(ft_chk_point))
        res = get_predictions(trainer, test_loader, model, device)
        
        with open(f'{exp_dir}/cv_{icv}.pkl', 'wb') as f:
            pickle.dump(res, f)
