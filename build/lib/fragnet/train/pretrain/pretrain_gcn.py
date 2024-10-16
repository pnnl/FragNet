from gcn import FragNetPreTrain
from dataset import load_data_parts
from data import mask_atom_features
import torch.nn as nn
from utils import EarlyStopping
import torch
from data import collate_fn
from torch.utils.data import DataLoader
from features import atom_list_one_hot
from tqdm import tqdm
import pickle
import os
from sklearn.model_selection import train_test_split

def load_ids(fn):

    if not os.path.exists('gcn_output/train_ids.pkl'):

        train_ids, test_ids = train_test_split(fn, test_size=.2)
        test_ids, val_ids = train_test_split(test_ids, test_size=.5)

        with open('gcn_output/train_ids.pkl', 'wb') as f:
            pickle.dump(train_ids, f)
        with open('gcn_output/val_ids.pkl', 'wb') as f:
            pickle.dump(val_ids, f)
        with open('gcn_output/test_ids.pkl', 'wb') as f:
            pickle.dump(test_ids, f)

    else:
        with open('gcn_output/train_ids.pkl', 'rb') as f:
            train_ids = pickle.load(f)
        with open('gcn_output/val_ids.pkl', 'rb') as f:
            val_ids = pickle.load(f)
        with open('gcn_output/test_ids.pkl', 'rb') as f:
            test_ids = pickle.load(f)

    return train_ids, val_ids, test_ids


def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in train_loader:
        # batch = data.to(device)
        mask_atom_features(batch)
        for k,v in batch.items():
            batch[k] = batch[k].to(device)
        optimizer.zero_grad()
        out = model(batch)
        labels = batch['x_atoms'][:, :len(atom_list_one_hot)].argmax(1)
        
        loss = loss_fn(out, labels)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
    return total_loss / len(train_loader.dataset)

def validate(loader, model, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        
        for batch in loader:
            for k,v in batch.items():
                batch[k] = batch[k].to(device)

            out = model(batch)
            labels = batch['x_atoms'][:, :len(atom_list_one_hot)].argmax(1)

            loss = loss_fn(out, labels)
            total_loss += loss.item()
        return total_loss / len(loader.dataset)



if __name__ == '__main__':

    files = os.listdir('pretrain_data/')
    fn = sorted([ int(i.split('.pkl')[0].strip('train')) for i in files if i.endswith('.pkl')])
    train_ids, val_ids,test_ids = load_ids(fn)
    train_dataset = load_data_parts('pretrain_data', 'train', include=train_ids)
    val_dataset = load_data_parts('pretrain_data', 'train', include=val_ids)    

    train_loader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=512, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=256, shuffle=False, drop_last=False)

    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    model_pretrain = FragNetPreTrain()
    model_pretrain.to(device)
    
    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model_pretrain.parameters(), lr = 1e-4 )
    chkpoint_name='pt_gcn.pt'
    early_stopping = EarlyStopping(patience=500, verbose=True, chkpoint_name=chkpoint_name)



    for epoch in tqdm(range(500)):

        train_loss = train(model_pretrain, train_loader, optimizer, device)
        val_loss = validate(val_loader, model_pretrain, device)
        print(train_loss, val_loss)


        early_stopping(val_loss, model_pretrain)

        if early_stopping.early_stop:
            print("Early stopping")
            break
