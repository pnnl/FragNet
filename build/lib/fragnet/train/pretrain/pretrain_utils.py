import pandas as pd
import torch

class Trainer:
    def __init__(self, loss_fn=None):
        self.loss_fn = loss_fn

    # single output regression
    def train(self, model, loader, optimizer, device):
        model.train()
        total_loss = 0
        for batch in loader:
            for k,v in batch.items():
                batch[k] = batch[k].to(device)
            optimizer.zero_grad()
            bond_length_pred, bond_angle_pred, dihedral_angle_pred, graph_rep = model(batch)
            bond_length_true = batch['bnd_lngth']
            bond_angle_true = batch['bnd_angl']
            dihedral_angle_true = batch['dh_angl']
            E = batch['y']

            loss_lngth = self.loss_fn(bond_length_pred, bond_length_true)
            loss_angle = self.loss_fn(bond_angle_pred, bond_angle_true)
            loss_lngth = self.loss_fn(dihedral_angle_pred, dihedral_angle_true)
            loss_E = self.loss_fn(graph_rep.view(-1), E)
            loss = loss_lngth + loss_angle + loss_lngth + loss_E
            
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


                bond_length_pred, bond_angle_pred, dihedral_angle_pred, graph_rep = model(batch)
                bond_length_true = batch['bnd_lngth']
                bond_angle_true = batch['bnd_angl']
                dihedral_angle_true = batch['dh_angl']
                E = batch['y']

                loss_lngth = self.loss_fn(bond_length_pred, bond_length_true)
                loss_angle = self.loss_fn(bond_angle_pred, bond_angle_true)
                loss_lngth = self.loss_fn(dihedral_angle_pred, dihedral_angle_true)
                loss_E = self.loss_fn(graph_rep.view(-1), E)
                loss = loss_lngth + loss_angle + loss_lngth + loss_E

                total_loss += loss.item()
            return total_loss / len(loader.dataset)
        

def load_prop_data(args):
    
    fs = []
    for f in args.pretrain.prop_files:
        fs.append(pd.read_csv(f))
    fs = pd.concat(fs, axis=0)
    fs.reset_index(drop=True, inplace=True)

    fs = fs.loc[:, ['smiles'] + args.pretrain.props]

    prop_dict = dict(zip(fs.smiles, fs.iloc[:, 1:].values.tolist()))

    return prop_dict, fs

def add_props_to_ds(ds, prop_dict):
    for d in ds:
        smiles = d.smiles
        props = prop_dict[smiles]
        d.y = torch.tensor([props], dtype=torch.float)