
from torch_geometric.data import DataLoader
from hyperopt import STATUS_OK
from dataset import load_pickle_dataset
from torch.utils.data import DataLoader
from data import collate_fn
import torch.nn as nn
import os
import torch
from omegaconf import OmegaConf
import argparse
from utils import TrainerFineTune as Trainer
import torch.optim.lr_scheduler as lr_scheduler
from ray import tune

RESULTS_PATH = './'

def trainModel(params):
    
    exp_dir = args.exp_dir
    n_classes_pt = args.pretrain.n_classes
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'


    pt_chkpoint_name = args.pretrain.chkpoint_name #'exps/pt_rings'
    from gat2 import FragNetFineTune
    
    model = FragNetFineTune(n_classes=args.finetune.model.n_classes, 
                            num_layer=params['num_layer'], 
                            drop_ratio=params['drop_ratio'])
    trainer = Trainer(target_type=args.finetune.target_type)


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
        
        
    args.finetune.train.path = os.path.join(RESULTS_PATH, args.finetune.train.path)
    args.finetune.val.path = os.path.join(RESULTS_PATH, args.finetune.val.path)
    args.finetune.test.path = os.path.join(RESULTS_PATH, args.finetune.test.path)
      
    train_dataset2 = load_pickle_dataset(args.finetune.train.path, args.finetune.train.name)
    val_dataset2 = load_pickle_dataset(args.finetune.val.path, args.finetune.val.name)
    test_dataset2 = load_pickle_dataset(args.finetune.test.path, args.finetune.test.name) #'finetune_data/pnnl_exp'

    train_loader = DataLoader(train_dataset2, collate_fn=collate_fn, batch_size=params['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset2, collate_fn=collate_fn, batch_size=params['batch_size'], shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset2, collate_fn=collate_fn, batch_size=params['batch_size'], shuffle=False, drop_last=False)

    model.to(device);



    if args.finetune.loss == 'mse':
        loss_fn = nn.MSELoss()
    elif args.finetune.loss == 'cel':
        loss_fn = nn.CrossEntropyLoss()
        
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4 ) # before 1e-4
    if args.finetune.use_schedular:
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    else:
        scheduler=None
        
    for epoch in range(args.finetune.n_epochs):

        train_loss = trainer.train(model=model, loader=train_loader, optimizer=optimizer, scheduler=scheduler, loss_fn=loss_fn, device=device)
        val_loss, _, _ = trainer.test(model=model, loader=val_loader, device=device)
        print(train_loss, val_loss)
    
    try:
        mse, true, pred = trainer.test(model=model, loader=val_loader, device=device)
        return {'score':mse, 'status':STATUS_OK}

    except:
        return {'score':1000, 'status':STATUS_OK}
    
    
    
parser = argparse.ArgumentParser()
parser.add_argument('--config', help="configuration file *.yml", type=str, required=False, default='exps/ft/esol2/None/no_pt/config.yaml')
args = parser.parse_args('')


if args.config:  # args priority is higher than yaml
    opt = OmegaConf.load(args.config)
    OmegaConf.resolve(opt)
    args=opt
    
    
args.finetune.n_epochs = 20
search_space = { 
    "num_layer": tune.choice([3,4,5,6,7,8]),
    "drop_ratio": tune.choice([0.1, 0.15, 0.2, 0.25, .3]),
    "batch_size": tune.choice([8, 16, 32, 64]),
}

tuner = tune.Tuner(trainModel, param_space=search_space,
                  tune_config=tune.TuneConfig(num_samples=100))
results = tuner.fit()
print(results.get_best_result(metric="score", mode="min").config)