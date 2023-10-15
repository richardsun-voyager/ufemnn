import torch
import torch.nn as nn
import numpy as np
from utils import length_to_mask

class CNN_Model(nn.Module):
    def __init__(self, config):
        super(CNN_Model, self).__init__()

        self.embeddings = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.cnn = nn.Conv1d(config.embedding_dim, config.hidden_dim, kernel_size=3, padding=2, stride=1, bias=False)
        self.final = nn.Linear(config.hidden_dim, config.label_dim, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(config.dropout)
        self.hidden_dim = config.hidden_dim
        self.label_dim = config.label_dim

        self.embeddings.weight.data.normal_(0, 0.1)
        self.final.weight.data.normal_(0, 0.1)

        self.device = config.device
        self.sqrt_d = np.sqrt(self.hidden_dim)
        
    def forward(self, seq_ids, seq_lengths):
        '''
        Args:
            seq_ids: word indices, batch_size, max_len, Long Tensor
            seq_lengths: lengths of sentences, batch_size, Long Tensor
        '''
        seq_embs = self.embeddings(seq_ids) / self.sqrt_d     # [batch_size, max_len, emb_dim]
        batch_size, max_len, embedding_dim = seq_embs.size()
        seq_embs = self.dropout(seq_embs)       

        seq_embs = seq_embs.transpose(1, 2)
        hidden_reprs = torch.relu(self.cnn(seq_embs)) / self.sqrt_d
        final_vecs = hidden_reprs.sum(2)
        scores = self.final(final_vecs)
        if self.label_dim == 1:
            probs = self.sigmoid(scores)
        else:
            probs = self.softmax(scores)

        return probs, scores
    