import torch
import numpy as np
from sklearn.metrics import mean_squared_error
from rdkit import Chem
from sklearn.metrics import roc_auc_score
from scipy.special import softmax
import torch.nn.functional as F
import torch.nn as nn


            

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, chkpoint_name = 'gnn_best.pt' ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.chkpoint_name = chkpoint_name

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.chkpoint_name)
        self.val_loss_min = val_loss
        
        
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





class Trainer:
    def __init__(self, target_pos=None, target_type='clsf', loss_fn=None,
                n_multi_task_heads=0):
        self.target_pos = target_pos
        self.loss_fn = loss_fn

        if target_type=='clsf':
            self.train = self.train_clsf
            self.validate = self.validate_clsf
        elif target_type=='regr':
            self.train = self.train_regr
            self.validate = self.validate_regr
        elif target_type=='regr1':
            self.train = self.train_regr1
            self.validate = self.validate_regr1
        elif target_type=='regr0':
            self.train = self.train
            self.validate = self.validate
        
        elif target_type=='clsf_ms':
            self.train = self.train_clsf_multi_task
            self.validate = self.validate_clsf_multi_task
            self.n_multi_task_heads = n_multi_task_heads
        
    
    # single output regression
    def train(self, model, loader, optimizer, device):
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
        return total_loss / len(loader.dataset)

    def validate(self, loader, model, device):
        model.eval()
        total_loss = 0
        with torch.no_grad():

            for batch in loader:
                for k,v in batch.items():
                    batch[k] = batch[k].to(device)

                out = model(batch).view(-1,)
                # labels = batch['x_atoms'][:, :len(atom_list_one_hot)].argmax(1)
                labels = batch['y']

                loss = self.loss_fn(out, labels)
                # loss = torch.nn.BCEWithLogitsLoss()(out.to(torch.float32), data.y.to(torch.float32))
                total_loss += loss.item()
            return total_loss / len(loader.dataset)
        
        
    # classification
    def train_clsf(self, model, loader, optimizer, device):
        model.train()
        total_loss = 0
        for batch in loader:
            for k,v in batch.items():
                batch[k] = batch[k].to(device)
            optimizer.zero_grad()
            out = model(batch)
            labels = batch['y'][:, self.target_pos].long()

            loss = self.loss_fn(out, labels)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        return total_loss / len(loader.dataset)

    def validate_clsf(self, loader, model, device):
        model.eval()
        total_loss = 0
        with torch.no_grad():

            for batch in loader:
                for k,v in batch.items():
                    batch[k] = batch[k].to(device)

                out = model(batch)
                labels = batch['y'][:, self.target_pos].long()

                loss = self.loss_fn(out, labels)
                total_loss += loss.item()
            return total_loss / len(loader.dataset)

    # regression when we have to chose from a list of target values
    def train_regr(self, model, loader, optimizer, device):
        model.train()
        total_loss = 0
        for batch in loader:
            for k,v in batch.items():
                batch[k] = batch[k].to(device)
            optimizer.zero_grad()
            out = model(batch)
            labels = batch['y']

            loss = self.loss_fn(out, labels)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        return total_loss / len(loader.dataset)

    def validate_regr(self, loader, model, device):
        model.eval()
        total_loss = 0
        with torch.no_grad():

            for batch in loader:
                for k,v in batch.items():
                    batch[k] = batch[k].to(device)

                out = model(batch)
                labels = batch['y']

                loss = self.loss_fn(out, labels)
                total_loss += loss.item()
            return total_loss / len(loader.dataset)
        
        
        
    def train_regr1(self, model, loader, optimizer, device):
        model.train()
        total_loss = 0
        for batch in loader:
            for k,v in batch.items():
                batch[k] = batch[k].to(device)
            optimizer.zero_grad()
            out = model(batch)
            out = out.view(-1,)
            labels = batch['y'][:, self.target_pos]

            loss = self.loss_fn(out, labels)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        return total_loss / len(loader.dataset)

    def validate_regr1(self, loader, model, device):
        model.eval()
        total_loss = 0
        with torch.no_grad():

            for batch in loader:
                for k,v in batch.items():
                    batch[k] = batch[k].to(device)

                out = model(batch)
                out = out.view(-1,)
                labels = batch['y'][:, self.target_pos]

                loss = self.loss_fn(out, labels)
                total_loss += loss.item()
            return total_loss / len(loader.dataset)
        
        
    # multi task classification
    def train_clsf_multi_task(self, model, loader, optimizer, device):
        model.train()
        total_loss = 0
        for batch in loader:
            for k,v in batch.items():
                batch[k] = batch[k].to(device)
            optimizer.zero_grad()
            outs = model(batch)
            
            loss = 0
            for i in range(self.n_multi_task_heads):
                
                labels = batch['y'][:, i]
                loss_i = self.loss_fn(outs[i], labels)
                loss += loss_i
                
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        return total_loss / len(loader.dataset)
        
    def validate_clsf_multi_task(self, loader, model, device):
        model.eval()
        total_loss = 0
        with torch.no_grad():

            for batch in loader:
                for k,v in batch.items():
                    batch[k] = batch[k].to(device)
                    
                outs = model(batch)
                
                loss = 0
                for i in range(self.n_multi_task_heads):
                    labels = batch['y'][:, i]
                    loss_i = self.loss_fn(outs[i], labels)
                    loss += loss_i    
                
                total_loss += loss.item()
            return total_loss / len(loader.dataset)
        
        
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

    
    def train_regr(self, model, loader, optimizer, scheduler, device, val_loader):
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
        #         pred = output.reshape(64,)
                pred = output

                target += list(data['y'].cpu().detach().numpy().ravel() )
                predicted += list(pred.cpu().detach().numpy().ravel() )
        mse = mean_squared_error(target, predicted)

        return mse, np.array(target), np.array(predicted)

    def train_clsf(self, model, loader, optimizer, scheduler, device, val_loader):
        model.train()
        total_loss = 0
        for batch in loader:
            for k,v in batch.items():
                batch[k] = batch[k].to(device)
            optimizer.zero_grad()
            out = model(batch)
            labels = batch['y'].view(-1,).long()

            loss = self.loss_fn(out, labels)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()

        if scheduler:
            val_loss = self.validate(model, val_loader, device)
            scheduler.step(val_loss)
            # scheduler.step()

        return total_loss / len(loader.dataset)
    

    def train_clsf_bce(self, model, loader, optimizer, scheduler, device, val_loader):
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
            loss.backward()
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

    # multi task classification
    def train_clsf_multi_task(self, model, loader, optimizer, scheduler, device, val_loader):
        model.train()
        total_loss = 0
        for batch in loader:
            for k,v in batch.items():
                batch[k] = batch[k].to(device)
            optimizer.zero_grad()
            outs = model(batch)

            is_valid = batch['y'] >-.5
            pred = outs[is_valid].float()
            target = batch['y'][is_valid].float()

            loss = F.binary_cross_entropy_with_logits(
                pred,target,reduction="sum")
                
            loss.backward()
            total_loss += loss.item()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # adding gradient clipping
            optimizer.step()

        if scheduler:
            val_loss = self.validate(model, val_loader, device)
            scheduler.step(val_loss)

        return total_loss / len(loader.dataset)
        
    def validate_clsf_multi_task(self, model, loader, device):
        model.eval()
        total_loss = 0
        with torch.no_grad():

            for batch in loader:
                for k,v in batch.items():
                    batch[k] = batch[k].to(device)
                    
                outs = model(batch)
                is_valid = batch['y'] >-.5
                pred = outs[is_valid].float()
                target = batch['y'][is_valid].float()

                loss = F.binary_cross_entropy_with_logits(
                    pred,target,reduction="sum")
                
                total_loss += loss.item()
            return total_loss / len(loader.dataset)
            
        
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

    
    def test_clsf_multi_task(self, model, loader, device):
    

        model.eval()
        with torch.no_grad():
            predicted=[]
            targets=[]
            for batch in loader:
                for k,v in batch.items():
                    batch[k] = batch[k].to(device)

                outputs = model(batch)
                probs = torch.sigmoid(outputs.float()).view(-1, outputs.size(-1))

                target_ = batch['y'].detach().cpu().numpy().tolist()
                probs_ =  probs.detach().cpu().numpy().tolist()

                targets.extend(target_)
                predicted.extend(probs_)
                
            
        agg_auc = self.get_roc_auc(targets, predicted)
        
        return -agg_auc, targets, predicted
    