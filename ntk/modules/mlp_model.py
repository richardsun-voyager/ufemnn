import torch
import torch.nn as nn
import numpy as np
from utils import length_to_mask

class MLP_Model(nn.Module):
    def __init__(self, config):
        super(MLP_Model, self).__init__()

        self.embeddings = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.hidden = nn.Linear(config.embedding_dim, config.hidden_dim, bias=False)
        self.final = nn.Linear(config.hidden_dim, config.label_dim, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(config.dropout)
        self.hidden_dim = config.hidden_dim
        self.label_dim = config.label_dim
        self.activation = config.activation

        self.embeddings.weight.data.normal_(0, 0.1)
        self.hidden.weight.data.normal_(0, 0.1)
        self.final.weight.data.normal_(0, config.final_init)

        self.device = config.device
        
        
    def forward(self, seq_ids, seq_lengths):
        '''
        Args:
            seq_ids: word indices, batch_size, max_len, Long Tensor
            seq_lengths: lengths of sentences, batch_size, Long Tensor
        '''
        seq_embs = self.embeddings(seq_ids) / np.sqrt(self.hidden_dim)    # [batch_size, max_len, emb_dim]
        batch_size, max_len, embedding_dim = seq_embs.size()
        seq_embs = self.dropout(seq_embs)       
        hidden_reprs = self.hidden(seq_embs)    # [batch_size, max_len, hidden_dim] 
        if self.activation == 'tanh':
            hidden_reprs = torch.tanh(hidden_reprs) / np.sqrt(self.hidden_dim)
        elif self.activation == 'relu':
            hidden_reprs = torch.relu(hidden_reprs) / np.sqrt(self.hidden_dim)
        elif self.activation == 'gelu':
            hidden_reprs = torch.nn.functional.gelu(hidden_reprs) / np.sqrt(self.hidden_dim)
        elif self.activation == 'silu':
            # swish, sigmoid-weighted linear unit
            hidden_reprs = hidden_reprs * torch.sigmoid(hidden_reprs) / np.sqrt(self.hidden_dim)
            # hidden_reprs = torch.nn.SiLU(hidden_reprs) / np.sqrt(self.hidden_dim)

        masks = length_to_mask(seq_lengths).expand(self.hidden_dim, batch_size, max_len)
        masks = masks.transpose(0, 1).transpose(1, 2)
        hidden_reprs = hidden_reprs * masks

        outputs = self.final(hidden_reprs.sum(1))
        if self.label_dim == 1:
            probs = self.sigmoid(outputs)
        else:
            probs = self.softmax(outputs)
        return probs, outputs
    