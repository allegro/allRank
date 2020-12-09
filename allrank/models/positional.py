"""
Code in this file was adapted from "The Annotated Transformer" by Harvard NLP.
http://nlp.seas.harvard.edu/2018/04/03/attention.html
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from allrank.config import PositionalEncoding


class FixedPositionalEncoding(nn.Module):
    """
    Class implementing fixed positional encodings.

    Fixed positional encodings up to max_len position are computed once during object construction.
    """
    def __init__(self, d_model: int, max_len=5000):
        """
        :param d_model: dimensionality of the embeddings
        :param max_len: maximum length of the sequence
        """
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
        """
        Forward pass through the FixedPositionalEncoding.
        :param x: input of shape [batch_size, slate_length, d_model]
        :param mask: padding mask of shape [batch_size, slate_length]
        :param indices: original item ranks used in positional encoding, shape [batch_size, slate_length]
        :return: output of shape [batch_size, slate_length, d_model]
        """
        padded_indices = indices.masked_fill(mask, self.padding_idx)
        padded_indices[padded_indices > self.padding_idx] = self.padding_idx
        x = math.sqrt(self.pe.shape[1]) * x + self.pe[padded_indices, :]  # type: ignore
        return x


class LearnedPositionalEncoding(nn.Module):
    """
    Class implementing learnable positional encodings.
    """
    def __init__(self, d_model, max_len=5000):
        """
        :param d_model: dimensionality of the embeddings
        :param max_len: maximum length of the sequence
        """
        super().__init__()

        self.pe = nn.Embedding(max_len + 1, d_model, padding_idx=-1)

    def forward(self, x, mask, indices):
        """
        Forward pass through the LearnedPositionalEncoding.
        :param x: input of shape [batch_size, slate_length, d_model]
        :param mask: padding mask of shape [batch_size, slate_length]
        :param indices: original item ranks used in positional encoding, shape [batch_size, slate_length]
        :return: output of shape [batch_size, slate_length, d_model]
        """
        padded_indices = indices.masked_fill(mask, self.pe.padding_idx)
        padded_indices[padded_indices > self.pe.padding_idx] = self.pe.padding_idx
        x = math.sqrt(self.pe.embedding_dim) * x + self.pe(padded_indices)
        return x


def _make_positional_encoding(d_model: int, positional_encoding: Optional[PositionalEncoding]):
    """
    Helper function for instantiating positional encodings classes.
    :param d_model: dimensionality of the embeddings
    :param positional_encoding: config.PositionalEncoding object containing PE config
    :return: positional encoding object of given variant
    """
    if positional_encoding is None:
        return None
    elif positional_encoding.strategy == "fixed":
        return FixedPositionalEncoding(d_model, max_len=positional_encoding.max_indices)
    elif positional_encoding.strategy == "learned":
        return LearnedPositionalEncoding(d_model, max_len=positional_encoding.max_indices)
    else:
        raise ValueError("Invalid positional encoding type: {}".format(positional_encoding.strategy))
