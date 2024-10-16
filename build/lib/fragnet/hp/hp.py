
from torch_geometric.data import DataLoader
import random
import hyperopt
from hyperopt import fmin, hp, Trials, STATUS_OK
import time
from dataset import load_pickle_dataset
from torch.utils.data import DataLoader
from data import collate_fn
from gat import FragNet
import torch.nn as nn
from gat import FragNetPreTrain
from utils import EarlyStopping
import numpy as np
from utils import test_fn
from dataset import load_data_parts
import os
import torch
from torch_scatter import scatter_add

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
    
    
    class FragNetFineTune(nn.Module):
    
        def __init__(self):
            super(FragNetFineTune, self).__init__()

            self.pretrain = FragNet(num_layer=6, drop_ratio= params['d1'] )
            self.lin1 = nn.Linear(128*2, int(params['f2']))
            self.lin2 = nn.Linear(int(params['f2']), int(params['f3']))
            self.out = nn.Linear(int(params['f3']), 1)
            self.dropout = nn.Dropout(p= params['d2'] ) 
            self.activation = nn.ReLU()


        def forward(self, batch):

            x_atoms, x_frags = self.pretrain(batch)

            x_frags_pooled = scatter_add(src=x_frags, index=batch['frag_batch'], dim=0)
            x_atoms_pooled = scatter_add(src=x_atoms, index=batch['batch'], dim=0)

            cat = torch.cat((x_atoms_pooled, x_frags_pooled), 1)
            x = self.dropout(cat)
                
            x = self.lin1(x)
            x = self.activation(x)
            x = self.dropout(x)
            
            x = self.lin2(x)
            x = self.activation(x)
            x = self.dropout(x)
            
            x = self.out(x)

            return x

    
    
    train_loader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=params['bs'], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=128, shuffle=False, drop_last=False)

    model_pretrain = FragNetPreTrain();
    model_pretrain.to(device);
    model_pretrain.load_state_dict(torch.load('pt.pt'))
    model = FragNetFineTune()
    model.to(device);
    loss_fn = nn.MSELoss()
    chkpoint_name = 'hp2.pt'
    

    model, optimizer = get_optimizer(model, freeze_pt_weights=freeze_pt_weights, lr=1e-5)
    early_stopping = EarlyStopping(patience=20, verbose=True, chkpoint_name=chkpoint_name)

    for epoch in range(n_epochs):

        res = []
        model.train()
        total_loss = 0
        for batch in train_loader:
            for k,v in batch.items():
                batch[k] = batch[k].to(device)
            optimizer.zero_grad()
            out = model(batch).view(-1,)
            loss = loss_fn(out, batch['y'])
            loss.backward()
            total_loss += loss.item()
            optimizer.step()


        try:
            val_loss, _, _ = test_fn(val_loader, model, device)
            res.append(val_loss)
            print("val mse: ", val_loss)

            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break
        except:
            print('val loss cannot be calculated')
            pass

    try:
        model.load_state_dict(torch.load(chkpoint_name))
        test_mse, test_t, test_p = test_fn(val_loader, model, device)
        return {'loss':test_mse, 'status':STATUS_OK}

    except:
        return {'loss':1000, 'status':STATUS_OK}

if __name__ == '__main__':

    dataset_name='moleculenet'
    dataset_subset = 'esol'
    freeze_pt_weights=False
    # add_pt_weights=True
    seed = None
    n_epochs = 100

    if dataset_name == 'moleculenet':
        train_dataset = load_pickle_dataset(f'{dataset_name}/{dataset_subset}', f'train_{str(seed)}')
        val_dataset = load_pickle_dataset(f'{dataset_name}/{dataset_subset}', f'val_{str(seed)}')
        test_dataset = load_pickle_dataset(f'{dataset_name}/{dataset_subset}', f'test_{str(seed)}')
    elif 'pnnl' in dataset_name:
        train_dataset = load_data_parts(path=dataset_name, name='train')
        val_dataset = load_data_parts(path=dataset_name, name='val')
        test_dataset = load_data_parts(path=dataset_name, name='test')


    space = {
            'f1': hp.quniform('f1', 32, 320,  32),
            'f2': hp.quniform('f2', 32, 320,  32),
            'f3': hp.quniform('f3', 32, 320,  32),
            'd1': hp.uniform('d1', 0,1),
            'd2': hp.uniform('d2', 0,1),
            'bs' : hp.choice('bs', [16, 32, 128]),
            }

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    trials = Trials()
    st_time = time.time()
    best = fmin(trainModel, space, algo=hyperopt.tpe.suggest, max_evals=100, trials=trials)

    end_time = time.time()
    fo = open("res2.txt", "w")
    fo.write(repr(best))
    fo.close()

    print( (end_time-st_time)/3600 )