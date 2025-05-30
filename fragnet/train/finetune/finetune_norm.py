import torch
from dataset import load_pickle_dataset
import torch.nn as nn
from utils import EarlyStopping
import torch
from data import collate_fn
from torch.utils.data import DataLoader
import argparse
from trainer_dta import TrainerFineTune as Trainer
import numpy as np
from omegaconf import OmegaConf
import pickle
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter

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



def save_predictions(trainer, loader, model, exp_dir, device, save_name='test_res', loss_type='mse', seed=123):
    score, true, pred = trainer.test(model=model, loader=loader, device=device)
    smiles = [i.smiles for i in loader.dataset]

    if loss_type=='mse':
        print(f'{save_name} rmse: ', score**.5)
        res = {'acc': score**.5, 'true': true, 'pred': pred, 'smiles': smiles}
    elif loss_type=='cel' or loss_type=='bce':
        # score = roc_auc_score(true, pred[:,1])
        print(f'{save_name} auc: ', score)
        res = {'acc': score, 'true': true, 'pred': pred, 'smiles': smiles}

    with open(f"{exp_dir}/{save_name}_{seed}.pkl", 'wb') as f:
        pickle.dump(res,f )




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="configuration file *.yml", type=str, required=False, default='config.yaml')
    args = parser.parse_args()

    if args.config: 
        opt = OmegaConf.load(args.config)
        OmegaConf.resolve(opt)
        opt.update(vars(args))
        args=opt

    seed_everything(args.seed)
    exp_dir = args['exp_dir']
    
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    print(args.model_version)
    writer = SummaryWriter(exp_dir+'/runs')    


    if args.model_version == 'gat':
        from gat import FragNetFineTune
        print('loaded from gat')
        model = FragNetFineTune(n_classes=args.finetune.model.n_classes)
        trainer = Trainer(target_type=args.finetune.target_type)

    elif args.model_version == 'gcn2':
        print ('importing model gcn')
        from gcn2 import FragNetFineTune
        model = FragNetFineTune(n_classes=args.finetune.model.n_classes,
                                atom_features=args.atom_features, 
                                frag_features=args.frag_features, 
                                edge_features=args.edge_features,
                                num_layer=args.finetune.model.num_layer, 
                                drop_ratio=args.finetune.model.drop_ratio,
                                emb_dim=args.finetune.model.emb_dim,
                                h1=args.finetune.model.h1,
                                h2=args.finetune.model.h2,
                                h3=args.finetune.model.h3,
                                h4=args.finetune.model.h4,
                                act=args.finetune.model.act,
                                fthead=args.finetune.model.fthead
                            )
        trainer = Trainer(target_type=args.finetune.target_type)
        
    elif args.model_version=='gat2':
        from gat2 import FragNetFineTune
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
        
        print('loaded from gat2')

    elif args.model_version=='gat2_lite':
        from gat2_lite import FragNetFineTune
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
        
        print('loaded from gat2_list')


    elif args.model_version=='gat2_edge':
        from gat2_edge import FragNetFineTune
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
        
        print('loaded from gat2')

        
    elif args.model_version=='gat2_transformer':
        from gat2 import FragNetFineTuneTransformer
        model = FragNetFineTuneTransformer(n_classes=args.finetune.model.n_classes, 
                            atom_features=args.atom_features, 
                            frag_features=args.frag_features, 
                            edge_features=args.edge_features,
                        num_layer=args.finetune.model.num_layer, 
                        drop_ratio=args.finetune.model.drop_ratio,
                                h1=args.finetune.model.h1, 
                                num_heads=args.finetune.model.num_heads, 
                                emb_dim=args.finetune.model.emb_dim)
        
        trainer = Trainer(target_type=args.finetune.target_type)
        
        
    elif args.model_version=='gat2_transformer2':
        from gat2 import FragNetFineTuneTransformer2
        model = FragNetFineTuneTransformer2(n_classes=args.finetune.model.n_classes, 
                        num_layer=args.finetune.model.num_layer, 
                        drop_ratio=args.finetune.model.drop_ratio,
                                h1=args.finetune.model.h1, 
                                num_heads=args.finetune.model.num_heads, 
                                emb_dim=args.finetune.model.emb_dim)
        
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
        model.pretrain.load_state_dict(modelpt.pretrain.state_dict())
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
        

    labels = [i.y for i in train_dataset2]
    label_mean, label_sdev = np.mean(labels), np.std(labels)

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
    save_predictions(trainer=trainer, loader=val_loader, model=model, exp_dir=exp_dir, device=device,  save_name='val_res', loss_type=args.finetune.loss, seed=args.seed)
    save_predictions(trainer=trainer, loader=test_loader, model=model, exp_dir=exp_dir, device=device, save_name='test_res', loss_type=args.finetune.loss, seed=args.seed)
    

