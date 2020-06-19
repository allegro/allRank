"""
Code in this file was adapted from "The Annotated Transformer" by Harvard NLP.
http://nlp.seas.harvard.edu/2018/04/03/attention.html
"""

import copy
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from allrank.config import PositionalEncoding
from allrank.models.positional import _make_positional_encoding

from allrank.models.positional import FixedPositionalEncoding, LearnedPositionalEncoding


def clones(module, N):
    """
    Creation of N identical layers.
    :param module: module to clone
    :param N: number of copies
    :return: nn.ModuleList of module copies
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    """
    Stack of Transformer encoder blocks with positional encoding.
    """
    def __init__(self, layer, output_layer, N, position, d_model, auxiliary_output, div_output):
        """
        :param layer: single building block to clone
        :param N: number of copies
        :param position: positional encoding module
        """
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.output_layers = clones(output_layer, N)
        self.norm = LayerNorm(layer.size)
        self.position = position
        self.auxiliary_output = auxiliary_output
        self.div_output = div_output
        if self.div_output:
            self.joint = nn.Linear(2, 1, bias=False)
            self.div_proj = nn.Linear(d_model, d_model)
        if self.auxiliary_output:
            if self.auxiliary_output == "fixed":
                self.pe = FixedPositionalEncoding(d_model)
            elif self.auxiliary_output == "learned":
                self.pe = LearnedPositionalEncoding(d_model)
            # elif self.auxiliary_output == "fixed_random":
            #     self.pe = RandomStaticPositionalEncoding(d_model)
            elif self.auxiliary_output == "identity":
                self.pe = lambda x, y, z: x
            else:
                raise RuntimeError("aux output strategy should be either 'fixed' or 'learned' or 'identity' but was {}".format(auxiliary_output))

    def forward(self, x, mask, indices):
        """
        Forward pass through each block of the Transformer.
        :param x: input of shape [batch_size, slate_length, input_dim]
        :param mask: padding mask of shape [batch_size, slate_length]
        :param indices: original item ranks used in positional encoding, shape [batch_size, slate_length]
        :return: output of shape [batch_size, slate_length, output_dim]
        """
        if self.position:
            x = self.position(x, mask, indices)
        mask = mask.unsqueeze(-2)
        if self.auxiliary_output:
            outputs = []
            next_x = x
            for layer, output_layer in zip(self.layers, self.output_layers):
                next_x = layer(next_x, mask)
                normed = self.norm(next_x)
                scored = output_layer(normed)
                _, indices = scored.sort(descending=True, dim=-1)
                next_x = self.pe(next_x, mask.squeeze(1), indices)
                outputs.append(scored)
            return outputs
        else:
            for layer in self.layers:
                x = layer(x, mask)
            normed_x = self.norm(x)
            scores = self.output_layers[-1](normed_x)
            if self.div_output:

                _, indices = scores.sort(descending=True, dim=-1)
                indices = torch.unsqueeze(indices, -1).repeat_interleave(normed_x.shape[-1], -1)
                x_sorted = torch.gather(normed_x, dim=1, index=indices)
                if "proj" in self.div_output:
                    x_sorted = self.div_proj(x_sorted)
                    # option - add projection here
                cdist = 1 - cosine_pairwise(x_sorted)
                # print("cdist")
                # print(cdist[0])
                distances_to_above = torch.tril(cdist)

                # print("distances_to_above")
                # print(distances_to_above[0])
                # print(distances_to_above[0][0])
                # print(distances_to_above[0][1])
                if "max" in self.div_output:
                    agg_distance_to_above, _ = distances_to_above.max(dim=2)
                else:
                    agg_distance_to_above, _ = distances_to_above.mean(dim=2)
                # print("agg_distance_to_above")
                # print(agg_distance_to_above[0])
                # print(agg_distance_to_above[0][0])
                # print(scores.shape)
                # print(max_distance_to_above.shape)
                agg_distance_to_above = agg_distance_to_above
                scores = self.joint(torch.stack((scores, agg_distance_to_above), dim=2)).squeeze()
                print(self.joint.weight)
                # print(scores.shape)
            return scores

    def score(self, x):
        if self.auxiliary_output:
            r = self.output_layers[-1].score(x[-1])
            return r
        else:
            r = self.output_layers[-1].score(x)
            return r

def cosine_pairwise(x):
    x = x.permute((1, 2, 0))
    cos_sim_pairwise = F.cosine_similarity(x, x.unsqueeze(1), dim=-2)
    cos_sim_pairwise = cos_sim_pairwise.permute((2, 0, 1))
    return cos_sim_pairwise

class LayerNorm(nn.Module):
    """
    Layer normalization module.
    """
    def __init__(self, features, eps=1e-6):
        """
        :param features: shape of normalized features
        :param eps: epsilon used for standard deviation
        """
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))  # type: ignore
        self.b_2 = nn.Parameter(torch.zeros(features))  # type: ignore
        self.eps = eps

    def forward(self, x):
        """
        Forward pass through the layer normalization.
        :param x: input of shape [batch_size, slate_length, input_dim]
        :return: normalized input of shape [batch_size, slate_length, output_dim]
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    Residual connection followed by layer normalization.
    Please not that for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        """
        :param size: number of input/output features
        :param dropout: dropout probability
        """
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        Forward pass through the sublayer connection module, applying the residual connection to any sublayer with the same size.
        :param x: input of shape [batch_size, slate_length, input_dim]
        :param sublayer: layer through which to pass the input prior to applying the sum
        :return: output of shape [batch_size, slate_length, output_dim]
        """
        return x + self.dropout(
            sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """
    Single Transformer encoder block made of self-attention and feed-forward layers with residual connections.
    """
    def __init__(self, size, self_attn, feed_forward, dropout):
        """
        :param size: input/output size of the encoder block
        :param self_attn: self-attention layer
        :param feed_forward: feed-forward layer
        :param dropout: dropout probability
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """
        Forward pass through the encoder block.
        :param x: input of shape [batch_size, slate_length, self.size]
        :param mask: padding mask of shape [batch_size, slate_length]
        :return: output of shape [batch_size, slate_length, self.size]
        """
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


def attention(query, key, value, mask=None, dropout=None):
    """
    Basic function for "Scaled Dot Product Attention" computation.
    :param query: query set of shape [batch_size, slate_size, n_attention_heads, attention_dim]
    :param key: key set of shape [batch_size, slate_size, n_attention_heads, attention_dim]
    :param value: value set of shape [batch_size, slate_size, n_attention_heads, attention_dim]
    :param mask: padding mask of shape [batch_size, slate_length]
    :param dropout: dropout probability
    :return: attention scores of shape [batch_size, slate_size, n_attention_heads, attention_dim]
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 1, float("-inf"))

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    Multi-headed attention block.
    """
    def __init__(self, h, d_model, dropout=0.1):
        """
        :param h: number of attention heads
        :param d_model: input/output dimensionality
        :param dropout: dropout probability
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        Forward pass through the multi-head attention block.
        :param query: query set of shape [batch_size, slate_size, self.d_model]
        :param key: key set of shape [batch_size, slate_size, self.d_model]
        :param value: value set of shape [batch_size, slate_size, self.d_model]
        :param mask: padding mask of shape [batch_size, slate_length]
        :return: output of shape [batch_size, slate_size, self.d_model]
        """
        if mask is not None:
            # same mask applied to all h heads
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [linear(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for linear, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    """
    Feed-forward block.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        :param d_model: input/output dimensionality
        :param d_ff: hidden dimensionality
        :param dropout: dropout probability
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass through the feed-forward block.
        :param x: input of shape [batch_size, slate_size, self.d_model]
        :return: output of shape [batch_size, slate_size, self.d_model]
        """
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


def make_transformer(output_layer, N=6, d_ff=2048, h=8, dropout=0.1, n_features=136,
                     positional_encoding: Optional[PositionalEncoding] = None,
                     aux_output = None, div_output= None):
    """
    Helper function for instantiating Transformer-based Encoder.
    :param N: number of Transformer blocks
    :param d_ff: hidden dimensionality of the feed-forward layer in the Transformer block
    :param h: number of attention heads
    :param dropout: dropout probability
    :param n_features: number of input/output features of the feed-forward layer
    :param positional_encoding: config.PositionalEncoding object containing PE config
    :return: Transformer-based Encoder with given hyperparameters
    """
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, n_features, dropout)

    ff = PositionwiseFeedForward(n_features, d_ff, dropout)
    position = _make_positional_encoding(n_features, positional_encoding)
    return Encoder(EncoderLayer(n_features, c(attn), c(ff), dropout), output_layer, N, position,
                   n_features, aux_output, div_output)
