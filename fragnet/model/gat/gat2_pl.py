from pytorch_lightning.callbacks import LearningRateMonitor
import pytorch_lightning as pl
from pytorch_lightning import loggers
import torch
import torch.nn.functional as F
from gat2 import FragNetFineTune
from pytorch_lightning import LightningModule, Trainer
from dataset import load_pickle_dataset
from torch.utils.data import DataLoader
import argparse
from omegaconf import OmegaConf
from torch_geometric.nn.norm import BatchNorm
from data import collate_fn
import math
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

def get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps,
        num_cycles = 0.5, last_epoch = -1):
    """
    Implementation by Huggingface:
    https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/optimization.py

    Create a schedule with a learning rate that decreases following the values
    of the cosine function between the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just
            decrease from the max value to 0 following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return max(1e-6, float(current_step) / float(max(1, num_warmup_steps)))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

class FragNetPL(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()

        self.model = FragNetFineTune(n_classes=args.finetune.model.n_classes, 
                            atom_features=args.atom_features, 
                            frag_features=args.frag_features, 
                            edge_features=args.edge_features,
                            num_layer=args.finetune.model.num_layer, 
                            drop_ratio=args.finetune.model.drop_ratio,
                                num_heads=args.finetune.model.num_heads, 
                                emb_dim=args.finetune.model.emb_dim,
                                h1=args.finetune.model.h1,
                                h2=args.finetune.model.h2,
                                h3=args.finetune.model.h3,
                                h4=args.finetune.model.h4,
                                act=args.finetune.model.act,
                                fthead=args.finetune.model.fthead
                                )

        self.args = args

    def forward(self, batch):
        return self.model(batch)

    def common_step(self, batch):
 
        y = self(batch)
        y_pred = batch['y']
        return y.reshape_as(y_pred), y_pred


    def training_step(self, batch, batch_idx):
        y, y_pred = self.common_step(batch)

        loss = F.mse_loss(y_pred, y)
        self.log('train_loss', loss, batch_size=args.finetune.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        y, y_pred = self.common_step(batch)
        val_loss = F.mse_loss(y_pred, y)

        self.log('val_loss', val_loss, batch_size=args.finetune.batch_size)

    def test_step(self, batch, batch_idx):
        y, y_pred = self.common_step(batch)
       
        test_loss = F.mse_loss(y_pred, y)
        self.log('test_mse', test_loss**.5, batch_size=args.finetune.batch_size)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.args.finetune.lr)
        return [optimizer]



if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="configuration file *.yml", type=str, required=False, default='config_exp1.yaml')
    args = parser.parse_args()


    if args.config:  # args priority is higher than yaml
        opt = OmegaConf.load(args.config)
        OmegaConf.resolve(opt)
        args=opt


    pl.seed_everything(args.seed)
    model = FragNetPL(args)


    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=100, verbose=False, mode="min")
    checkpoint = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min')

    trainer = Trainer(accelerator='auto', max_epochs=args.finetune.n_epochs, gradient_clip_val=1, devices=-1,
                        log_every_n_steps=1, callbacks=[early_stop_callback, checkpoint])


    train_dataset2 = load_pickle_dataset(args.finetune.train.path)
    val_dataset2 = load_pickle_dataset(args.finetune.val.path)
    test_dataset2 = load_pickle_dataset(args.finetune.test.path) #'finetune_data/pnnl_exp'

    train_loader = DataLoader(train_dataset2, collate_fn=collate_fn, batch_size=args.finetune.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset2, collate_fn=collate_fn, batch_size=64, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset2, collate_fn=collate_fn, batch_size=64, shuffle=False, drop_last=False)

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)