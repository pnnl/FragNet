import torch
from torch import nn
import numpy as np
from .drug_encoder import Encoder_MultipleLayers

torch.manual_seed(1)
np.random.seed(1)

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):

        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    """Construct the embeddings from protein/target, position embeddings.
    """
    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_rate):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_size, hidden_size)

        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings



class transformer(nn.Sequential):
    def __init__(self):
        super(transformer, self).__init__()
        input_dim_drug = 25
        transformer_emb_size_drug = 128
        transformer_dropout_rate = 0.1
        transformer_n_layer_drug = 8
        transformer_intermediate_size_drug = 512
        transformer_num_attention_heads_drug = 8
        transformer_attention_probs_dropout = 0.1
        transformer_hidden_dropout_rate = 0.1
        max_position_size = 1000
        self.emb = Embeddings(input_dim_drug,
                         transformer_emb_size_drug,
                         max_position_size,
                         transformer_dropout_rate)

        self.encoder = Encoder_MultipleLayers(transformer_n_layer_drug,
                                         transformer_emb_size_drug,
                                         transformer_intermediate_size_drug,
                                         transformer_num_attention_heads_drug,
                                         transformer_attention_probs_dropout,
                                         transformer_hidden_dropout_rate)
    def forward(self, e, e_mask):
        ex_e_mask = e_mask.unsqueeze(1).unsqueeze(2)
        ex_e_mask = (1.0 - ex_e_mask) * -10000.0

        emb = self.emb(e)
        encoded_layers = self.encoder(emb.float(), ex_e_mask.float())
        return encoded_layers[:, 0]



class DTAModel(nn.Module):
    def __init__(self, drug_model):
        super().__init__()
        self.drug_model = drug_model # FragNetFineTune()
        self.target_model = transformer()
        self.fc1 = nn.Linear(256+128, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, batch):

        drug_enc = self.drug_model(batch)

        tokens = batch['protein']
        padding_mask = ~tokens.eq(0)*1
        
        target_enc = self.target_model(tokens, padding_mask)
        cat = torch.cat((drug_enc, target_enc), 1)

        out = self.fc1(cat)
        out = self.fc2(out)

        return out
        
        
class DTAModel2(nn.Module):
    def __init__(self, drug_model):
        super().__init__()
        self.drug_model = drug_model # FragNetFineTune()
        self.fc1 = nn.Linear(256+300, 128)
        self.fc2 = nn.Linear(128, 1)

        num_features = 25
        prot_emb_dim = 300
        self.in_channels = 1000
        n_filters = 32
        kernel_size = 8
        prot_output_dim=300

        self.embedding_xt = nn.Embedding(num_features + 1, prot_emb_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=self.in_channels, out_channels=n_filters, kernel_size=kernel_size)
        intermediate_dim = prot_emb_dim - kernel_size + 1
        self.fc1_xt_dim = n_filters*intermediate_dim
        self.fc1_xt = nn.Linear(self.fc1_xt_dim, prot_output_dim)

    def forward(self, batch):

        drug_enc = self.drug_model(batch)

        tokens = batch['protein']
        tokens = tokens.reshape(-1, self.in_channels)

        embedded_xt = self.embedding_xt(tokens)
        conv_xt = self.conv_xt_1(embedded_xt)
        # flatten
        xt = conv_xt.view(-1, self.fc1_xt_dim)
        xt = self.fc1_xt(xt)


        cat = torch.cat((drug_enc, xt), 1)

        out = self.fc1(cat)
        out = self.fc2(out)

        return out
        
        
        
