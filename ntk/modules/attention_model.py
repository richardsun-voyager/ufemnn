import torch
import torch.nn as nn
import numpy as np
import math
from torch.autograd import Variable
from utils import length_to_mask

import math
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout=0.5, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)/20
        return self.dropout(x)

class Attention_Model(nn.Module):
    def __init__(self, config):
        super(Attention_Model, self).__init__()
        self.max_length = 512
        self.embeddings = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.pos_embeddings = PositionalEncoding(config.embedding_dim)
        self.final = nn.Linear(config.hidden_dim, config.label_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(config.dropout)
        self.hidden_dim = config.hidden_dim
        self.label_dim = config.label_dim

        self.embeddings.weight.data.normal_(0, 0.1)
        self.final.weight.data.normal_(0, 0.1)
        
        if "dense" in config.model_name:
            self.model_name = 'dense'
            self.hidden = nn.Linear(config.embedding_dim, self.max_length)
            self.hidden.weight.data.normal_(0, 0.1)
        elif "random" in config.model_name:
            self.model_name = 'random'
            self.random_attn = nn.Parameter(torch.randn())
            self.random_attn.data.normal_(0, 0.1)
        else:
            self.model_name = 'attention'
            self.linear_q = nn.Linear(config.embedding_dim, config.hidden_dim, bias=False)
            self.linear_k = nn.Linear(config.embedding_dim, config.hidden_dim, bias=False)
            self.linear_v = nn.Linear(config.embedding_dim, config.hidden_dim, bias=False)
            self.linear_q.weight.data.normal_(0, 0.1)
            self.linear_k.weight.data.normal_(0, 0.1)
            self.linear_v.weight.data.normal_(0, 0.1)

        self.device = config.device
        self.sqrt_d = np.sqrt(self.hidden_dim)

    def forward(self, seq_ids, seq_lengths):
        seq_embs = self.embeddings(seq_ids)     # [batch_size, max_len, emb_dim]
        seq_pos_embs = self.pos_embeddings(seq_embs)
        batch_size, max_len, embedding_dim = seq_embs.size()
        keys = seq_embs
        queries = seq_pos_embs
        values = seq_embs

        masks = length_to_mask(seq_lengths).float()
        masks = masks.unsqueeze(-1).matmul(masks.unsqueeze(1))

        attention_scores = torch.bmm(queries, keys.transpose(1, 2))/self.sqrt_d
        attention_scores = attention_scores.masked_fill(~masks.bool().to(self.device), -100000)
        attn = self.softmax(attention_scores) 
        
        hidden_vecs = torch.bmm(attn, values).squeeze(1)
        
        outputs = self.final(hidden_vecs.sum(1))/self.sqrt_d
        if self.label_dim == 1:
            probs = self.sigmoid(outputs)
        else:
            probs = self.softmax(outputs)
        return probs, outputs

  