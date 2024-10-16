import torch
import optuna
from torch_geometric.data import DataLoader
import random
import time
from fragnet.dataset.dataset import load_pickle_dataset
from torch.utils.data import DataLoader
from fragnet.dataset.data import collate_fn
import torch.nn as nn
from fragnet.model.gat.gat2 import FragNet
from fragnet.train.utils import EarlyStopping
import numpy as np
from fragnet.train.utils import test_fn
from fragnet.dataset.dataset import load_data_parts
import os
import torch
from torch_scatter import scatter_add
from omegaconf import OmegaConf
import argparse
from fragnet.train.utils import TrainerFineTune as Trainer
import torch.optim.lr_scheduler as lr_scheduler
from fragnet.dataset.data import collate_fn
from fragnet.dataset.data import collate_fn
import matplotlib.pyplot as plt
from fragnet.model.gat.gat2 import FragNetFineTune
import pytorch_lightning as pl

def seed_everything(seed: int):
    import random, os

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def trainModel(trial):

    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    num_heads = args.finetune.model.num_heads
    num_layer = args.finetune.model.num_layer

    drop_ratio = trial.suggest_categorical('drop_ratio', [0.0, 0.1, 0.2, 0.3])

    fthead = args.finetune.model.fthead

    if fthead == 'FTHead3':
        h1 = trial.suggest_int('h1', 64, 2048, step=64)
        h2=trial.suggest_int('h2', 64, 2048, step=64)
        h3=trial.suggest_int('h3', 64, 2048, step=64)
        h4=trial.suggest_int('h4', 64, 2048, step=64)

    elif fthead == 'FTHead4':
        h1 = trial.suggest_int('h1', 64, 2048, step=64)
        h2, h3, h4 = None, None, None

    
    act = trial.suggest_categorical("act", ['relu','silu','gelu','celu','selu','rrelu','relu6','prelu','leakyrelu'])
    batch_size  = trial.suggest_categorical('batch_size', [16,32,64,128])
    lr = args.finetune.lr
   

    pt_chkpoint_name = args.pretrain.chkpoint_name #'exps/pt_rings'
    if args.model_version == 'gat2':
        from fragnet.model.gat.gat2 import FragNetFineTune
    elif args.model_version == 'gat2_lite':
        from fragnet.model.gat.gat2_lite import FragNetFineTune

    model = FragNetFineTune(n_classes=args.finetune.model.n_classes,
                                atom_features=args.atom_features, 
                                frag_features=args.frag_features, 
                                edge_features=args.edge_features,
                                num_heads = num_heads,
                                num_layer= num_layer,
                                drop_ratio= drop_ratio,
                                h1=h1, 
                                h2=h2, 
                                h3=h3, 
                                h4=h4,
                                act=act,
                                fthead=fthead
                            )
        
   

    if pt_chkpoint_name:

        from fragnet.model.gat.gat2_pretrain import FragNetPreTrain
        modelpt = FragNetPreTrain(
                            atom_features=args.atom_features, 
                            frag_features=args.frag_features, 
                            edge_features=args.edge_features,
                            num_layer=args.pretrain.num_layer, 
                            drop_ratio=args.pretrain.drop_ratio,
                            num_heads=args.pretrain.num_heads, 
                            emb_dim=args.pretrain.emb_dim)
        modelpt.load_state_dict(torch.load(pt_chkpoint_name, map_location=torch.device(device)))


        print('loading pretrained weights')
        model.pretrain.load_state_dict(modelpt.pretrain.state_dict())
        print('weights loaded')
    else:
        print('no pretrained weights')


    trainer = Trainer(target_type=args.finetune.target_type)

    train_dataset2 = load_pickle_dataset(args.finetune.train.path)
    val_dataset2 = load_pickle_dataset(args.finetune.val.path)


    train_loader = DataLoader(train_dataset2, collate_fn=collate_fn, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset2, collate_fn=collate_fn, batch_size=128, shuffle=False, drop_last=False)
   
    model.to(device);

    optimizer = torch.optim.Adam(model.parameters(), lr = lr ) # before 1e-4
    if args.finetune.use_schedular:
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, cooldown=0)
    else:
        scheduler=None
        
    scheduler=None
    
    ft_chk_point = args.chkpt
    print("checkpoint name: ", ft_chk_point)
    early_stopping = EarlyStopping(patience=args.finetune.es_patience, verbose=True, chkpoint_name=ft_chk_point)

    for epoch in range(args.finetune.n_epochs):

        train_loss = trainer.train(model=model, loader=train_loader, optimizer=optimizer, scheduler=scheduler, device=device, val_loader=val_loader)
        val_loss, _, _ = trainer.test(model=model, loader=val_loader, device=device)
        print(train_loss, val_loss)


        trial.report(val_loss, epoch)
        if args.prune == 1:
            if trial.should_prune():
                raise optuna.TrialPruned()

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    
    try:
        model.load_state_dict(torch.load(ft_chk_point))
        mse, true, pred = trainer.test(model=model, loader=val_loader, device=device)
        
        return mse

    except:
        return 1000.0
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="configuration file *.yml", type=str, required=False, default='exps/ft/esol2/config.yaml')
    parser.add_argument('--chkpt', help="checkpoint name", type=str, required=False, default='opt.pt')
    parser.add_argument('--direction', help="", type=str, required=False, default='minimize')
    parser.add_argument('--n_trials', help="", type=int, required=False, default=100)
    parser.add_argument('--embed_max', help="", type=int, required=False, default=512)
    parser.add_argument('--seed', help="", type=int, required=False, default=1)
    parser.add_argument('--ft_epochs', help="", type=int, required=False, default=100)
    parser.add_argument('--choose_random', help="", type=bool, required=False, default=False)
    parser.add_argument('--prune', help="", type=int, required=False, default=1)
    args = parser.parse_args()


    if args.config:  # args priority is higher than yaml
        opt = OmegaConf.load(args.config)
        OmegaConf.resolve(opt)

        opt.update(vars(args))
        args = opt

    seed_everything(args.seed)
    print('seed: ', args.seed)
    print('choose_random: ', args.choose_random)
    args.finetune.n_epochs = args.ft_epochs

    # for resuming 
    study_name = args.chkpt.replace('.pt', '').split('/')[-1]
    storage = args.chkpt.replace('.pt', '.db')
    study = optuna.create_study(direction = args.direction, study_name=study_name, storage=f'sqlite:///{storage}', load_if_exists=True)
    # for resuming

    study.optimize(trainModel, n_trials=args.n_trials, gc_after_trial=True)
    df_study = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
    df_name = args.chkpt.replace('.pt', '.csv')
    df_study.to_csv(df_name)    

    print("best params:")
    print(study.best_params)

    