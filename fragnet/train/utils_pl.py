import torch
import numpy as np
from sklearn.metrics import mean_squared_error
from rdkit import Chem
from sklearn.metrics import roc_auc_score
from scipy.special import softmax
import torch.nn.functional as F
import torch.nn as nn


            
        
def test_fn(loader, model, device):
    
    model.eval()
    with torch.no_grad():
        target, predicted = [], []
        for data in loader:
            for k,v in data.items():
                data[k] = data[k].to(device)
            
            output = model(data)
    #         pred = output.reshape(64,)
            pred = output

            target += list(data['y'].cpu().numpy().ravel() )
            predicted += list(pred.cpu().numpy().ravel() )
    mse = mean_squared_error(target, predicted)
    return mse, np.array(target), np.array(predicted)


def compute_nll_loss(prediction, target):
    lprobs = F.log_softmax(prediction.float(), dim=-1)
    lprobs = lprobs.view(-1, lprobs.size(-1))
    # target = target.view(-1)
    loss = F.nll_loss(
        lprobs,
        target,
        reduction="sum")
    return loss

def compute_bce_loss(prediction, target):

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    is_valid = target > -.5
    loss_mat = criterion(prediction, target)  # shape = [N, C]
    loss_mat = torch.where(is_valid, loss_mat, torch.zeros_like(loss_mat))  # shape = [N, C]
    loss = torch.sum(loss_mat)/torch.sum(is_valid)
    return loss


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

    
    def train_regr(self, model, loader, optimizer, scheduler, device, val_loader, fabric):
        model.train()
        total_loss = 0
        for batch in loader:
            for k,v in batch.items():
                batch[k] = batch[k].to(device)
            optimizer.zero_grad()
            out = model(batch).view(-1,)
            labels = batch['y']

            loss = self.loss_fn(out, labels)
            fabric.backward(loss)
            total_loss += loss.item()
            optimizer.step()
    
        if scheduler:
            val_loss = self.validate(model, val_loader, device)
            scheduler.step()

        return total_loss / len(loader.dataset)

    def validate_regr(self, model,loader, device):
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

        
    def test_regr(self, model, loader, device):
    
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

    def train_clsf(self, model, loader, optimizer, scheduler, device, val_loader, fabric):
        model.train()
        total_loss = 0
        for batch in loader:
            for k,v in batch.items():
                batch[k] = batch[k].to(device)
            optimizer.zero_grad()
            out = model(batch)
            labels = batch['y'].view(-1,).long()

            loss = self.loss_fn(out, labels)
            fabric.backward(loss)
            total_loss += loss.item()
            optimizer.step()

        if scheduler:
            val_loss = self.validate(model, val_loader, device)
            scheduler.step(val_loss)

        return total_loss / len(loader.dataset)
    

    def train_clsf_bce(self, model, loader, optimizer, scheduler, device, val_loader, fabric):
        model.train()
        total_loss = 0
        for batch in loader:
            for k,v in batch.items():
                batch[k] = batch[k].to(device)
            
            out = model(batch)
            labels = batch['y'].view(out.shape)

            is_valid = batch['y'] > -0.5  # shape = [N, C]
            loss_mat = self.loss_fn(out, labels)

            loss_mat = torch.where(
            is_valid, loss_mat, torch.zeros_like(loss_mat))  # shape = [N, C]

            optimizer.zero_grad()
            loss = torch.sum(loss_mat)/torch.sum(is_valid)
            fabric.backward(loss)
            total_loss += loss.item()
            optimizer.step()

        if scheduler:
            val_loss = self.validate(model, val_loader, device)
            scheduler.step(val_loss)

        return total_loss / len(loader.dataset)

    # when nll loss is used
    def validate_clsf(self, model, loader, device):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            target, predicted = [], []

            for batch in loader:
                for k,v in batch.items():
                    batch[k] = batch[k].to(device)

                output = model(batch)

                pred = output.softmax(dim=1)

                target += list(batch['y'].cpu().numpy().ravel().astype(int) )
                predicted.extend(list(pred.cpu().numpy().tolist() ) )
        
            target = np.array(target)
            predicted = np.array(predicted)
            roc_auc = roc_auc_score(target, np.array(predicted)[:, 1])

            return -roc_auc
    
    # when bce loss is used
    def validate_clsf_bce(self, model, loader, device):
        model.eval()
        with torch.no_grad():
            target, predicted = [], []

            for batch in loader:
                for k,v in batch.items():
                    batch[k] = batch[k].to(device)

                pred = model(batch)
                target.append(batch['y'].view(pred.shape))
                predicted.append(pred)

   
            target = torch.cat(target, dim=0).cpu().numpy()
            predicted = torch.cat(predicted, dim=0).cpu().numpy()
            roc_list = []
            for i in range(target.shape[1]):
                # AUC is only defined when there is at least one positive data.
                if np.sum(target[:, i] == 1) > 0 and np.sum(target[:, i] == 0) > 0:
                    is_valid = target[:, i] > -.5
                    roc_list.append(roc_auc_score(
                        (target[is_valid, i]), predicted[is_valid, i]))
                

            roc_auc = sum(roc_list) / len(roc_list)
            return -roc_auc
    
    # when nll loss is used
    def test_clsf(self, model, loader, device):
    
        model.eval()
        with torch.no_grad():
            target, predicted = [], []
            for data in loader:
                for k,v in data.items():
                    data[k] = data[k].to(device)

                output = model(data)
                pred = output.softmax(dim=1)

                target += list(data['y'].cpu().numpy().ravel().astype(int) )
                predicted.extend(list(pred.cpu().numpy().tolist() ) )
        
        target = np.array(target)
        predicted = np.array(predicted)
        roc_auc = roc_auc_score(target, np.array(predicted)[:, 1])
        
        return -roc_auc, target, predicted
    
    # when bce loss is used
    def test_clsf_bce(self, model, loader, device):
    
        model.eval()
        with torch.no_grad():
            target, predicted = [], []
            for data in loader:
                for k,v in data.items():
                    data[k] = data[k].to(device)

                pred = model(data)

                target.append(data['y'].view(pred.shape))
                predicted.append(pred)

            target = torch.cat(target, dim=0).cpu().numpy()
            predicted = torch.cat(predicted, dim=0).cpu().numpy()
            roc_list = []
            for i in range(target.shape[1]):
                # AUC is only defined when there is at least one positive data.
                if np.sum(target[:, i] == 1) > 0 and np.sum(target[:, i] == 0) > 0:
                    is_valid = target[:, i] > -.5
                    roc_list.append(roc_auc_score(
                        (target[is_valid, i]), predicted[is_valid, i]))
        
        roc_auc = sum(roc_list) / len(roc_list)
        
        return -roc_auc, target, predicted
        

    def get_roc_auc(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        agg_auc_list=[]
        for i in range(y_true.shape[1]):
                # AUC is only defined when there is at least one positive data.
                if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
                    # ignore nan values
                    is_labeled = y_true[:, i] > -0.5
                    agg_auc_list.append(
                        roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])
                    )

        agg_auc = sum(agg_auc_list) / len(agg_auc_list)

        return agg_auc