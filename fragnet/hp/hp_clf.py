
from torch_geometric.data import DataLoader
import random
import hyperopt
from hyperopt import fmin, hp, Trials, STATUS_OK
import time
from dataset import load_pickle_dataset
from torch.utils.data import DataLoader
from data import collate_fn
import torch.nn as nn
from utils import EarlyStopping
import numpy as np
import os
import torch
from omegaconf import OmegaConf
import argparse
from utils import TrainerFineTune as Trainer
import torch.optim.lr_scheduler as lr_scheduler

def get_optimizer(model, freeze_pt_weights=False, lr=1e-4):
    
    if freeze_pt_weights:
        print('freezing pretrain weights')
        for name, param in model.named_parameters():
            if param.requires_grad and 'pretrain' in name:
                param.requires_grad = False

        non_frozen_parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(non_frozen_parameters, lr = lr)
    else:
        print('no freezing of the weights')
        optimizer = torch.optim.Adam(model.parameters(), lr = lr )

    return model, optimizer


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    

def trainModel(params):
    

    exp_dir = args.exp_dir
    n_classes_pt = args.pretrain.n_classes
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    pt_chkpoint_name = args.pretrain.chkpoint_name #'exps/pt_rings'

    from gat2 import FragNetFineTune    
    model = FragNetFineTune(n_classes=args.finetune.model.n_classes, 
                            num_layer=params['num_layer'], 
                            drop_ratio=params['drop_ratio'])


    if pt_chkpoint_name:
        model_pretrain = FragNetFineTune(n_classes_pt)
        model_pretrain.load_state_dict(torch.load(pt_chkpoint_name))
        state_dict_to_load={}
        for k,v in model.state_dict().items():
            
            if v.size() == model_pretrain.state_dict()[k].size():
                state_dict_to_load[k] = model_pretrain.state_dict()[k]
            else:
                state_dict_to_load[k] = v
            
        model.load_state_dict(state_dict_to_load)
        
    train_dataset2 = load_pickle_dataset(args.finetune.train.path, args.finetune.train.name)
    val_dataset2 = load_pickle_dataset(args.finetune.val.path, args.finetune.val.name)
    test_dataset2 = load_pickle_dataset(args.finetune.test.path, args.finetune.test.name) #'finetune_data/pnnl_exp'


    train_loader = DataLoader(train_dataset2, collate_fn=collate_fn, batch_size=params['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset2, collate_fn=collate_fn, batch_size=params['batch_size'], shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset2, collate_fn=collate_fn, batch_size=params['batch_size'], shuffle=False, drop_last=False)

    trainer = Trainer(target_type=args.finetune.target_type)


    model.to(device);
    ft_chk_point = f'{args.exp_dir}/fthp.pt'
    early_stopping = EarlyStopping(patience=100, verbose=True, chkpoint_name=ft_chk_point)


    if args.finetune.loss == 'mse':
        loss_fn = nn.MSELoss()
    elif args.finetune.loss == 'cel':
        loss_fn = nn.CrossEntropyLoss()
        
    optimizer = torch.optim.Adam(model.parameters(), lr = params['lr'] ) # before 1e-4
    if args.finetune.use_schedular:
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    else:
        scheduler=None
        
    for epoch in range(args.finetune.n_epochs):

        train_loss = trainer.train(model=model, loader=train_loader, optimizer=optimizer, scheduler=scheduler, loss_fn=loss_fn, device=device)
        val_loss, _, _ = trainer.test(model=model, loader=val_loader, device=device)
        print(train_loss, val_loss)
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    try:
        model.load_state_dict(torch.load(ft_chk_point))
        mse, true, pred = trainer.test(model=model, loader=val_loader, device=device)
        
        return {'loss':-mse, 'status':STATUS_OK}

    except:
        return {'loss':-100000, 'status':STATUS_OK}
    
    
parser = argparse.ArgumentParser()
parser.add_argument('--config', help="configuration file *.yml", type=str, required=False, default='config.yaml')
args = parser.parse_args()


if args.config:  # args priority is higher than yaml
    opt = OmegaConf.load(args.config)
    OmegaConf.resolve(opt)
    args=opt


space = {
        'num_layer': hp.choice('num_layer', [3, 4, 5,6,7,8]),
        'lr': hp.choice('lr', [1e-3, 1e-4, 1e-6]),
        'drop_ratio': hp.choice('drop_ratio', [0.15, 0.2, 0.3, 0.5]),
        'batch_size' : hp.choice('batch_size', [8, 16, 32, 64, 128]),
        }


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

trials = Trials()
st_time = time.time()
best = fmin(trainModel, space, algo=hyperopt.rand.suggest, max_evals=25, trials=trials)

end_time = time.time()
fo = open(f"{args.exp_dir}/res2.txt", "w")
fo.write(repr(best))
fo.close()

print( (end_time-st_time)/3600 )
