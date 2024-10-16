import torch
from dataset import load_pickle_dataset
import torch.nn as nn
from utils import EarlyStopping
import torch
from data import collate_fn_cdrp as collate_fn
from torch.utils.data import DataLoader
import argparse
from trainer_cdrp import TrainerFineTune as Trainer
import numpy as np
from omegaconf import OmegaConf
import pickle
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from gat2 import FragNet, FTHead3
from torch_scatter import scatter_add
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
    

def get_train_stats(ds, exp_dir):
    y_old = np.array([i.y.item() for i in ds])
    mean = np.mean(y_old)
    sdev = np.std(y_old)
    with open(f'{exp_dir}/train_stats.pkl', 'wb') as f:
        pickle.dump({'mean':mean, 'sdev':sdev}, f)
    return mean, sdev

def scale_y(ds, mean, sdev):
    y_old = np.array([i.y.item() for i in ds])
    y = (y_old - mean)/sdev
    for i, d in enumerate(ds):
        d.y = torch.tensor([y[i]], dtype=torch.float )



def save_predictions(trainer, loader, model, exp_dir, device, save_name='test_res', loss_type='mse', seed=123,
                     label_mean=None, label_sdev=None):
    score, true, pred = trainer.test(model=model, loader=loader, device=device, label_mean=label_mean, label_sdev=label_sdev)
    smiles = [i.smiles for i in loader.dataset]

    if loss_type=='mse':
        print(f'{save_name} rmse: ', score**.5)
        res = {'acc': score**.5, 'true': true, 'pred': pred, 'smiles': smiles}
    elif loss_type=='cel' or loss_type=='bce':
        print(f'{save_name} auc: ', score)
        res = {'acc': score, 'true': true, 'pred': pred, 'smiles': smiles}

    with open(f"{exp_dir}/{save_name}_{seed}.pkl", 'wb') as f:
        pickle.dump(res,f )


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



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="configuration file *.yml", type=str, required=False, default='config.yaml')
    args = parser.parse_args()

    if args.config:  # args priority is higher than yaml
        opt = OmegaConf.load(args.config)
        OmegaConf.resolve(opt)
        opt.update(vars(args))
        args=opt

    seed_everything(args.seed)
    exp_dir = args['exp_dir']
    
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    print(args.model_version)
    writer = SummaryWriter(exp_dir+'/runs')



    
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
        
        
    pt_chkpoint_name = args.pretrain.chkpoint_name #'exps/pt_rings'
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

        
        print(f'loading pretrained weights from {pt_chkpoint_name}')
        model.drug_model.pretrain.load_state_dict(modelpt.pretrain.state_dict())
        print('weights loaded')
    else:
        print('no pretrained weights')



    train_dataset2 = load_pickle_dataset(args.finetune.train.path)
    val_dataset2 = load_pickle_dataset(args.finetune.val.path)
    test_dataset2 = load_pickle_dataset(args.finetune.test.path) #'finetune_data/pnnl_exp'

    train_loader = DataLoader(train_dataset2, collate_fn=collate_fn, batch_size=args.finetune.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset2, collate_fn=collate_fn, batch_size=64, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset2, collate_fn=collate_fn, batch_size=64, shuffle=False, drop_last=False)

    

    model.to(device);
    ft_chk_point = args.finetune.chkpoint_name
    early_stopping = EarlyStopping(patience=args.finetune.es_patience, verbose=True, chkpoint_name=ft_chk_point)


    if args.finetune.loss == 'mse':
        loss_fn = nn.MSELoss()
    elif args.finetune.loss == 'cel':
        loss_fn = nn.CrossEntropyLoss()
    else: loss_fn = None
        
    optimizer = torch.optim.Adam(model.parameters(), lr = args.finetune.lr ) # before 1e-4
    if args.finetune.use_schedular:
        scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=30)
    else:
        scheduler=None
        

    label_mean, label_sdev = None, None
    
    for epoch in range(args.finetune.n_epochs):

        train_loss = trainer.train(model=model, loader=train_loader, optimizer=optimizer, scheduler=scheduler, 
                                   device=device, val_loader=val_loader, label_mean=label_mean, label_sdev=label_sdev) 
        val_loss, _, _ = trainer.test(model=model, loader=val_loader, device=device, 
                                      label_mean=label_mean, label_sdev=label_sdev)

        print("epoch: ", epoch, train_loss, val_loss)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)

        if epoch%1==0:

            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break




    model.load_state_dict(torch.load(ft_chk_point))
    save_predictions(trainer=trainer, loader=val_loader, model=model, exp_dir=exp_dir, device=device,  save_name='val_res', loss_type=args.finetune.loss, seed=args.seed,
                     label_mean=label_mean, label_sdev=label_sdev)
    save_predictions(trainer=trainer, loader=test_loader, model=model, exp_dir=exp_dir, device=device, save_name='test_res', loss_type=args.finetune.loss, seed=args.seed,
                     label_mean=label_mean, label_sdev=label_sdev)
    

