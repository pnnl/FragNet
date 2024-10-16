import torch
import optuna
from torch_geometric.data import DataLoader
from dataset import load_pickle_dataset
from torch.utils.data import DataLoader
from data import collate_fn_cdrp as collate_fn
import torch.nn as nn
from gat2 import FragNet
from utils import EarlyStopping
import numpy as np
import torch
from torch_scatter import scatter_add
from omegaconf import OmegaConf
import argparse
from trainer_cdrp import TrainerFineTune as Trainer
import torch.optim.lr_scheduler as lr_scheduler
from gat2 import FragNet, FTHead3
from cdrp_model.model import CDRPModel


def seed_everything(seed: int):
    import random, os

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class FragNetFineTuneBase(nn.Module):
    
    def __init__(self, n_classes=1, atom_features=167, frag_features=167, edge_features=17, 
                 num_layer=4, num_heads=4, drop_ratio=0.15,
                h1=256, h2=256, h3=256, h4=256, act='celu',emb_dim=128, fthead='FTHead3'):
        super().__init__()
        

        self.pretrain = FragNet(num_layer=num_layer, drop_ratio=drop_ratio, 
                                num_heads=num_heads, emb_dim=emb_dim,
                                atom_features=atom_features, frag_features=frag_features, edge_features=edge_features)


        if fthead == 'FTHead1':
            self.fthead = FTHead1(n_classes=n_classes)
        elif fthead == 'FTHead2':
            print('using FTHead2' )
            self.fthead = FTHead2(n_classes=n_classes)
        elif fthead == 'FTHead3':
            print('using FTHead3' )
            self.fthead = FTHead3(n_classes=n_classes,
                             h1=h1, h2=h2, h3=h3, h4=h4,
                             drop_ratio=drop_ratio, act=act)
            
        elif fthead == 'FTHead4':
            print('using FTHead4' )
            self.fthead = FTHead4(n_classes=n_classes,
                             h1=h1, drop_ratio=drop_ratio, act=act)
        
                    
        
    def forward(self, batch):
        
        x_atoms, x_frags, x_edge, x_fedge = self.pretrain(batch)
        
        x_frags_pooled = scatter_add(src=x_frags, index=batch['frag_batch'], dim=0)
        x_atoms_pooled = scatter_add(src=x_atoms, index=batch['batch'], dim=0)
        
        cat = torch.cat((x_atoms_pooled, x_frags_pooled), 1)

            
        return cat





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
        from gat2 import FragNetFineTune
    elif args.model_version == 'gat2_lite':
        from gat2_lite import FragNetFineTune

   

    gat2 = FragNetFineTuneBase(n_classes=args.finetune.model.n_classes, 
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
                                fthead=args.finetune.model.fthead)
    
    model=CDRPModel(gat2, args.gene_dim, device)
    trainer = Trainer(target_type=args.finetune.target_type)

    if pt_chkpoint_name:

        from models import FragNetPreTrain
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
        model.drug_model.pretrain.load_state_dict(modelpt.pretrain.state_dict())
        print('weights loaded')
    else:
        print('no pretrained weights')


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

    label_mean, label_sdev = None, None
    for epoch in range(args.finetune.n_epochs):

        train_loss = trainer.train(model=model, loader=train_loader, optimizer=optimizer, scheduler=scheduler, device=device, val_loader=val_loader,
                                   label_mean=label_mean, label_sdev=label_sdev)
        val_loss, _, _ = trainer.test(model=model, loader=val_loader, device=device,
                                      label_mean=label_mean, label_sdev=label_sdev)
        print(train_loss, val_loss)


        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    try:
        model.load_state_dict(torch.load(ft_chk_point))
        mse, true, pred = trainer.test(model=model, loader=val_loader, device=device,
                                       label_mean=label_mean, label_sdev=label_sdev)
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

    # study = optuna.create_study(direction = args.direction)
    study.optimize(trainModel, n_trials=args.n_trials, gc_after_trial=True)

    df_study = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
    df_name = args.chkpt.replace('.pt', '.csv')
    df_study.to_csv(df_name)    

    print("best params:")
    print(study.best_params)

    