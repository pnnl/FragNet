
from torch_geometric.data import DataLoader
import hyperopt
from hyperopt import fmin, hp, Trials, STATUS_OK
import time
from dataset import load_pickle_dataset
from torch.utils.data import DataLoader
from data import collate_fn
from gat2 import FragNet
import torch.nn as nn
from utils import EarlyStopping
import numpy as np
import torch
from torch_scatter import scatter_add
from omegaconf import OmegaConf
import argparse
from utils import TrainerFineTune as Trainer
import torch.optim.lr_scheduler as lr_scheduler


def seed_everything(seed: int):
    import random, os

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(123)

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



class FTHead2(nn.Sequential):
    def __init__(self, input_dim=128, h1=128, h2=1024, h3=1024, 
                 h4=512, drop_ratio=.2, n_classes=1, act='relu'):
        super().__init__()


        self.dropout = nn.Dropout(p=drop_ratio)
        if act == 'relu':
            self.activation = nn.ReLU()
        elif act == 'silu':
            self.activation = nn.SiLU()
        elif act == 'gelu':
            self.activation = nn.GELU()
        elif act == 'celu':
            self.activation = nn.CELU()
        elif act == 'selu':
            self.activation = nn.SELU()
        elif act == 'rrelu':
            self.activation = nn.RReLU()
        elif act == 'relu6':
            self.activation = nn.ReLU6()
        elif act == 'prelu':
            self.activation = nn.PReLU()
        elif act == 'leakyrelu':
            self.activation = nn.LeakyReLU()
            
        self.hidden_dims =  [h1, h2, h3, h4]
        layer_size = len(self.hidden_dims) + 1
        dims = [input_dim*2] + self.hidden_dims + [n_classes]
        self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(layer_size)])
    

    def forward(self, enc):
  
        for i in range(0, len(self.predictor)-1):
            enc = self.activation(self.dropout(self.predictor[i](enc)))
        out = self.predictor[-1](enc)

        return out


class FragNetFineTune(nn.Module):
    
    def __init__(self, n_classes=1, num_layer=4, drop_ratio=0.15,
            num_heads=4, emb_dim=128, h1=None, h2=None, h3=None, h4=None,
            act=None):
        super().__init__()
        

        self.pretrain = FragNet(num_layer=num_layer, drop_ratio=drop_ratio, 
                                num_heads=num_heads, emb_dim=emb_dim)
        self.fthead = FTHead2(n_classes=n_classes,
                             h1=h1, h2=h2, h3=h3, h4=h4,
                             drop_ratio=drop_ratio, act=act)
                
            
        
    def forward(self, batch):
        
        x_atoms, x_frags, x_edge = self.pretrain(batch)
        
        x_frags_pooled = scatter_add(src=x_frags, index=batch['frag_batch'], dim=0)
        x_atoms_pooled = scatter_add(src=x_atoms, index=batch['batch'], dim=0)
        
        cat = torch.cat((x_atoms_pooled, x_frags_pooled), 1)
        x = self.fthead(cat)
        
        return x




def trainModel(params):


    exp_dir = args.exp_dir
    device = 'cpu'

    pt_chkpoint_name = args.pretrain.chkpoint_name #'exps/pt_rings'
    model = FragNetFineTune(n_classes=args.finetune.model.n_classes, 
                        drop_ratio = params['drop_ratio'],
                        h1 = int(params['h1']), 
                        h2 = int(params['h2']), 
                        h3 = int(params['h3']), 
                        h4 = int(params['h4']),
                        act = params['act'],
                        num_layer = int(params['num_layer']),
                        num_heads = int(params['num_heads'])

                       )

    if pt_chkpoint_name:

            from models import FragNetPreTrain
            modelpt = FragNetPreTrain(num_layer=args.pretrain.num_layer, 
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

    train_loader = DataLoader(train_dataset2, collate_fn=collate_fn, batch_size=int(params['batch_size']), shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset2, collate_fn=collate_fn, batch_size=32, shuffle=False, drop_last=False)

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
        
    scheduler=None
    
    ft_chk_point = f'ft_hp.pt'
    early_stopping = EarlyStopping(patience=args.finetune.es_patience, verbose=True, chkpoint_name=ft_chk_point)


        
    for epoch in range(50):

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
        
        return {'loss':mse, 'status':STATUS_OK}

    except:
        return {'loss':1000, 'status':STATUS_OK}
    
    
parser = argparse.ArgumentParser()
parser.add_argument('--config', help="configuration file *.yml", type=str, required=False, default='exps/ft/esol2/config.yaml')
args = parser.parse_args()


if args.config:  # args priority is higher than yaml
    opt = OmegaConf.load(args.config)
    OmegaConf.resolve(opt)
    args=opt


space = {
        'h1': hp.quniform('h1', 64, 2048,  64),
        'h2': hp.quniform('h2', 64, 2048,  64),
        'h3': hp.quniform('h3', 64, 2048,  64),
        'h4': hp.quniform('h4', 64, 2048,  64),
        'drop_ratio': hp.choice('drop_ratio', [0.1 , 0.15, 0.2 , 0.25]),
        'num_layer': hp.choice('num_layer', [3, 4, 5, 6]),
        'num_heads': hp.choice('num_heads', [4, 8]),
        'batch_size' : hp.quniform('batch_size', 16, 64, 8),
        'act' : hp.choice('act', ['relu','silu','gelu','celu','selu','rrelu',
                        'relu6','prelu','leakyrelu'])

        }

trials = Trials()
st_time = time.time()
best = fmin(trainModel, space=space, algo=hyperopt.tpe.suggest, max_evals=20, trials=trials)

end_time = time.time()
fo = open(f"{args.exp_dir}/hpopt.txt", "w")
fo.write(repr(best))
fo.close()
print( (end_time-st_time)/3600 )
