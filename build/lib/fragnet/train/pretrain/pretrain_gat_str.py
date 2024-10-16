from gat import FragNetPreTrain
from dataset import load_data_parts
from data import mask_atom_features
import torch.nn as nn
from utils import EarlyStopping
import torch
from data import collate_fn
from torch.utils.data import DataLoader
from features import atom_list_one_hot


"""
this is to pretrain on molecular fragments
"""
def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in train_loader:
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

    train_dataset = load_data_parts('pretrain_data', 'train', include=range(0,479))
    val_dataset = load_data_parts('pretrain_data', 'train', include=range(479,499))
    
    train_loader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=512, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=256, shuffle=False, drop_last=False)
    
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    model_pretrain = FragNetPreTrain()
    model_pretrain.to(device)
    
    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model_pretrain.parameters(), lr = 1e-4 )
    chkpoint_name='pt_gat.pt'
    early_stopping = EarlyStopping(patience=100, verbose=True, chkpoint_name=chkpoint_name)



    for epoch in range(500):

        train_loss = train(model_pretrain, train_loader, optimizer, device)
        val_loss = validate(val_loader, model_pretrain, device)
        print(train_loss, val_loss)


        early_stopping(val_loss, model_pretrain)

        if early_stopping.early_stop:
            print("Early stopping")
            break
