from dataset import load_pickle_dataset
from torch.utils.data import DataLoader
from data import collate_fn
from gat import FragNetFineTune
import torch.nn as nn
from gat import FragNetPreTrain
import torch
from utils import EarlyStopping
import numpy as np
from utils import test_fn
import argparse
import os
import pandas as pd


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

def save_preds(test_dataset, true, pred, exp_dir):

    smiles = [i.smiles for i in test_dataset]
    res= pd.DataFrame(np.column_stack([smiles, true, pred]), columns=['smiles', 'true', 'pred'])
    res.to_csv(os.path.join(exp_dir, 'test_predictions.csv'), index=False)
    
def train(train_loader, model, device, optimizer):
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
    return total_loss / len(train_loader.dataset)

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
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', help="", type=str,required=False, default='moleculenet')
    parser.add_argument('--dataset_subset', help="", type=str,required=False, default='esol')
    parser.add_argument('--seed', help="", type=int,required=False, default=None)
    parser.add_argument('--batch_size', help="saving file name", type=int,required=False, default=32)
    parser.add_argument('--checkpoint', help="checkpoint name", type=str,required=False, default='gnn.pt')
    parser.add_argument('--add_pt_weights', help="checkpoint name", type=bool,required=False, default=False)
    parser.add_argument('--freeze_pt_weights', help="checkpoint name", type=bool,required=False, default=False)
    parser.add_argument('--exp_dir', help="", type=str,required=False, default='exps/pnnl')
    args = parser.parse_args()

    dataset_name = args.dataset_name      # 'esol'
    seed = args.seed #   36
    dataset_subset =  args.dataset_subset.lower()#  'moleculenet'
    batch_size = args.batch_size
    exp_dir = args.exp_dir
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    print('dataset: ', dataset_name, 'subset: ', dataset_subset)


    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    if dataset_name == 'moleculenet':
        train_dataset = load_pickle_dataset(f'{dataset_name}/{dataset_subset}', f'train_{seed}')
        val_dataset = load_pickle_dataset(f'{dataset_name}/{dataset_subset}', f'val_{seed}')
        test_dataset = load_pickle_dataset(f'{dataset_name}/{dataset_subset}', f'test_{seed}')
    elif 'pnnl' in dataset_name:
        train_dataset = load_pickle_dataset(path=dataset_name, name='train')
        val_dataset = load_pickle_dataset(path=dataset_name, name='val')
        test_dataset = load_pickle_dataset(path=dataset_name, name='test')



    train_loader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=False, drop_last=False)

    if args.add_pt_weights:
        model_pretrain = FragNetPreTrain();
        model_pretrain.to(device);
        model_pretrain.load_state_dict(torch.load('pt.pt'))
    model = FragNetFineTune()
    model.to(device);
    loss_fn = nn.MSELoss()


    if args.add_pt_weights:
        print("adding pretrain weights to the model")
        model.pretrain.load_state_dict(model_pretrain.pretrain.state_dict())

    model, optimizer = get_optimizer(model, freeze_pt_weights=args.freeze_pt_weights, lr=1e-4)

    chkpoint_name= os.path.join(exp_dir, args.checkpoint)
    early_stopping = EarlyStopping(patience=200, verbose=True, chkpoint_name=chkpoint_name)


    for epoch in range(2000):

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



        val_loss, _, _ = test_fn(val_loader, model, device)
        res.append(val_loss)
        print("val mse: ", val_loss)

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break


    model.load_state_dict(torch.load(chkpoint_name))

    mse, true, pred = test_fn(test_loader, model, device)
    save_preds(test_dataset, true, pred, exp_dir)
    print("rmse: ", mse**.5)



