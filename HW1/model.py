from typing import Dict

import torch
import torch.nn as nn
from torch.nn import Embedding

class SeqClassifier(torch.nn.Module):
    
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
        max_len: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        
        # TODO: model architecture
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.max_len = max_len
        
        # self.encoder = torch.nn.RNN(
        # self.encoder = torch.nn.LSTM(
        self.encoder = torch.nn.GRU(
            input_size=embeddings.shape[1], 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            dropout=dropout, 
            bidirectional=bidirectional,
            batch_first=True)
        
        self.dropout = torch.nn.Dropout(dropout)
        self.linear_1 = nn.Linear(self.encoder_output_size, hidden_size)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.leaky_relu = nn.LeakyReLU()
        self.linear_2 = nn.Linear(hidden_size, num_class)

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        if self.bidirectional:
            return self.hidden_size * self.max_len * 2
        else:
            return self.hidden_size * self.max_len

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        x = self.embed(batch)
        x, _ = self.encoder(x)
        x = x.resize(x.size(0), self.encoder_output_size)
        x = self.dropout(x)
        x = self.linear_1(x)
        x = self.batch_norm(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x


class SeqTagger(SeqClassifier):
        
    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        if self.bidirectional:
            return self.hidden_size * 2
        else:
            return self.hidden_size
    
    def forward(self, batch, tags=None) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        x = self.embed(batch)
        x, _ = self.encoder(x)
        x = self.dropout(x)
        x = self.linear_1(x)
        x = self.batch_norm(x.permute([0,2,1]))
        x = self.leaky_relu(x.permute([0,2,1]))
        x = self.dropout(x)
        x = self.linear_2(x)
        return x.permute([0,2,1])