from utils import compute_bce_loss
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, roc_auc_score
import torch.nn as nn


class TrainerFineTune:
    def __init__(self, target_pos=None, target_type='regr',
                n_multi_task_heads=0):
        self.target_pos = target_pos
        if target_type=='clsf':
            
            self.train = self.train_clsf_bce
            self.validate = self.validate_clsf_bce
            self.test = self.test_clsf_bce
            self.loss_fn = compute_bce_loss

        elif target_type=='regr':
            self.train = self.train_regr
            self.validate = self.validate_regr
            self.test = self.test_regr
            self.loss_fn = nn.MSELoss()

            
        elif target_type=='clsf_ms':
            self.train = self.train_clsf_multi_task
            self.validate = self.validate_clsf_multi_task
            self.test = self.test_clsf_multi_task
            self.n_multi_task_heads = n_multi_task_heads

    
    def train_regr(self, model, loader, optimizer, scheduler, device, val_loader, label_mean, label_sdev):
        model.train()
        total_loss = 0
        for batch in loader:
            for k,v in batch.items():
                batch[k] = batch[k].to(device)
            optimizer.zero_grad()
            out = model(batch).view(-1,)
            labels = batch['y']

            loss = self.loss_fn(out, labels)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
    
        if scheduler:
            val_loss = self.validate(model, val_loader, device)
            scheduler.step()

        return total_loss / len(loader.dataset)

    def validate_regr(self, model,loader, device, label_mean, label_sdev):
        model.eval()
        total_loss = 0
        with torch.no_grad():

            for batch in loader:
                for k,v in batch.items():
                    batch[k] = batch[k].to(device)

                out = model(batch).view(-1,)
                labels = batch['y']
                loss = self.loss_fn(out, labels)
                total_loss += loss.item()
            return total_loss / len(loader.dataset)

        
    def test_regr(self, model, loader, device, label_mean, label_sdev):
    
        model.eval()
        with torch.no_grad():
            target, predicted = [], []
            for data in loader:
                for k,v in data.items():
                    data[k] = data[k].to(device)
                
                output = model(data).view(-1,)
                pred = output

                target += list(data['y'].cpu().detach().numpy().ravel() )
                predicted += list(pred.cpu().detach().numpy().ravel() )
        mse = mean_squared_error(target, predicted)

        return mse, np.array(target), np.array(predicted)

 