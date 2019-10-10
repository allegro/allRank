import math
from typing import Optional

import torch
import torch.nn as nn

from allrank.config import PositionalEncoding


class FixedPositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model: int, max_len=5000):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = torch.cat((pe, torch.zeros([1, d_model])))
        self.padding_idx = pe.size()[0] - 1
        self.register_buffer('pe', pe)

    def forward(self, x, mask, indices):
        padded_indices = indices.masked_fill(mask, self.padding_idx)
        padded_indices[padded_indices > self.padding_idx] = self.padding_idx
        x = math.sqrt(self.pe.shape[1]) * x + self.pe[padded_indices, :]
        return x


class LearnedPositionalEncoding(nn.Module):
    "Implement the Learned PE function."

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        self.pe = nn.Embedding(max_len + 1, d_model, padding_idx=-1)

    def forward(self, x, mask, indices):
        padded_indices = indices.masked_fill(mask, self.pe.padding_idx)
        padded_indices[padded_indices > self.pe.padding_idx] = self.pe.padding_idx
        x = math.sqrt(self.pe.embedding_dim) * x + self.pe(padded_indices)
        return x


def _make_positional_encoding(d_model: int, positional_encoding: Optional[PositionalEncoding]):
    if positional_encoding is None:
        return None
    elif positional_encoding.strategy == "fixed":
        return FixedPositionalEncoding(d_model, max_len=positional_encoding.max_indices)
    elif positional_encoding.strategy == "learned":
        return LearnedPositionalEncoding(d_model, max_len=positional_encoding.max_indices)
    else:
        raise ValueError("Invalid positional encoding type: {}".format(positional_encoding.strategy))
