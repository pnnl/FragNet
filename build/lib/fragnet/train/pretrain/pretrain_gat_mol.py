import torch.nn as nn
from utils import EarlyStopping
import torch
from data import collate_fn
from torch.utils.data import DataLoader
import argparse
import numpy as np
from omegaconf import OmegaConf
from utils import Trainer
from torch.utils.tensorboard import SummaryWriter
from dataset import LoadDataSets
from pretrain_utils import load_prop_data, add_props_to_ds


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

"""
This is to pretrain on molecular properties
"""


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="configuration file *.yml", type=str, required=False, default='config.yml')
    args = parser.parse_args()

    if args.config:  # args priority is higher than yaml
        opt = OmegaConf.load(args.config)
        OmegaConf.resolve(opt)
        args=opt

    writer = SummaryWriter(args.exp_dir)
    if args.model_version == 'gat':
        from gat import FragNetFineTune
    elif args.model_version == 'gat2':
        from gat2 import FragNetFineTune
       
    
    ds = LoadDataSets()

    train_dataset, val_dataset, test_dataset = ds.load_datasets(args)

    prop_dict, _ = load_prop_data(args)
    add_props_to_ds(train_dataset, prop_dict)
    add_props_to_ds(val_dataset, prop_dict)


    train_loader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=512, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=512, shuffle=False, drop_last=False)

    n_classes = args.pretrain.n_classes # 31 for nRings
    target_pos = args.pretrain.target_pos
    target_type = args.pretrain.target_type

    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    model = FragNetFineTune(n_classes=args.pretrain.n_classes, 
                            num_layer=args.pretrain.num_layer, 
                            drop_ratio=args.pretrain.drop_ratio)

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4 )
    chkpoint_name = args.pretrain.chkpoint_name #'pt_gat_nring.pt'
    early_stopping = EarlyStopping(patience=args.pretrain.n_epochs, verbose=True, chkpoint_name=chkpoint_name)

    if args.pretrain.loss == 'mse':
        loss_fn = nn.MSELoss()
    elif args.pretrain.loss == 'cel':
        loss_fn = nn.CrossEntropyLoss()
        
    pretrainer = Trainer(target_pos=target_pos, target_type=target_type, loss_fn=loss_fn)


    for epoch in range(args.pretrain.n_epochs):

        train_loss = pretrainer.train(model, train_loader, optimizer, device)
        val_loss = pretrainer.validate(val_loader, model, device)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break
