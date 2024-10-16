import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Sequential):
    def __init__(self, gene_dim=903, device='cuda'):
        self.device = device
        input_dim_gene = gene_dim
        hidden_dim_gene = 256
        mlp_hidden_dims_gene = [1024, 256, 64]
        super(MLP, self).__init__()
        layer_size = len(mlp_hidden_dims_gene) + 1
        dims = [input_dim_gene] + mlp_hidden_dims_gene + [hidden_dim_gene]
        self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(layer_size)])

    def forward(self, v):
        # predict
        v = v.float().to(self.device)
        for i, l in enumerate(self.predictor):
            v = F.relu(l(v))
        return v
        
        
class CDRPModel(nn.Module):
    def __init__(self, drug_model, gene_dim, device):
        super().__init__()
        self.drug_model = drug_model # FragNetFineTune()
        self.fc1 = nn.Linear(256+256, 128)
        self.fc2 = nn.Linear(128, 1)
        self.cell_model = MLP(gene_dim, device)


    def forward(self, batch):

        drug_enc = self.drug_model(batch)
        gene_expr = batch['gene_expr']
        cell_enc = self.cell_model(gene_expr)

        cat = torch.cat((drug_enc, cell_enc), 1)
        out = self.fc1(cat)
        out = self.fc2(out)
        return out
        
        
        




