from gcn import FragNetPreTrain
from dataset import load_data_parts
from data import mask_atom_features
import torch.nn as nn
from utils import EarlyStopping
import torch
from data import collate_fn
from torch.utils.data import DataLoader
from features import atom_list_one_hot
from sklearn.model_selection import train_test_split
import os
import pickle
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning import loggers
import torch.optim as optim


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


class FragNetModule(pl.LightningModule):

    def __init__(self, model):
        """
        Inputs:`
            model_name - Name of the model/CNN to run. Used for creating the model (see function below)
            model_hparams - Hyperparameters for the model, as dictionary.
            optimizer_name - Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
        """
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # Create model
        self.model = model


    def forward(self, batch):
        # Forward function that is run when visualizing the graph
      
        out = self.model(batch)
        labels = batch['x_atoms'][:, :len(atom_list_one_hot)].argmax(1)
        preds = out.argmax(1)
        return preds, labels

    def configure_optimizers(self):
        
        optimizer = optim.AdamW(self.parameters())

        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)


        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        out = self.model(batch)
        labels = batch['x_atoms'][:, :len(atom_list_one_hot)].argmax(1)
        
        loss = loss_fn(out, labels)

        self.log('train_loss', loss)
        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        # x, y, p, scaffold = batch
        out = self.model(batch)
        labels = batch['x_atoms'][:, :len(atom_list_one_hot)].argmax(1)
        loss = loss_fn(out, labels)
        
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        out = self.model(batch)
        labels = batch['x_atoms'][:, :len(atom_list_one_hot)].argmax(1)
        loss = loss_fn(out, labels)
        
        self.log('test_loss', loss)



if __name__ == "__main__":


    files = os.listdir('pretrain_data/')
    fn = sorted([ int(i.split('.pkl')[0].strip('train')) for i in files if i.endswith('.pkl')])

    train_ids, val_ids,test_ids = load_ids(fn)

    train_dataset = load_data_parts('pretrain_data', 'train', include=train_ids)
    val_dataset = load_data_parts('pretrain_data', 'train', include=val_ids)
    test_dataset = load_data_parts('pretrain_data', 'train', include=test_ids)

    train_loader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=512, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=256, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=256, shuffle=False, drop_last=False)

    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'


    model_pretrain = FragNetPreTrain()
    loss_fn = nn.CrossEntropyLoss()


    checkpoint_callback = ModelCheckpoint(
    save_weights_only=True, mode="min", monitor="val_loss",
    save_top_k=1,
    verbose=True,
    # dirpath=f'{args.exp_dir}/ckpt',
    dirpath=None,
    filename='model_best')
    # if args.trainer_version:
    trainer_version='tmp'

    logger = loggers.TensorBoardLogger(save_dir='./', version=trainer_version, name="lightning_logs")
    trainer = pl.Trainer(default_root_dir='./',                          # Where to save models
                    accelerator="gpu" if str(device).startswith("cuda") else "cpu",        
                    devices=1,                                                         
                    max_epochs=100,                 
                    callbacks=[checkpoint_callback, LearningRateMonitor("epoch")],                      
                    enable_progress_bar=True,
                    gradient_clip_val=10,
                    precision=16,
                    limit_train_batches=None,
                    limit_val_batches=None,
                    logger=logger
                    )                                                          
    trainer.logger._log_graph = True      
    trainer.logger._default_hp_metric = None


    model = FragNetModule(model_pretrain)
    trainer.fit(model, train_loader, val_loader)