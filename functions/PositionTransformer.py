"""
@description:
源自FCMSTrans（tf改torch）, 用于处理ESM输出: 121, 1280
"""

import torch
import torch.nn as nn
import math


def positional_encoding(position, d_model):
    angle_rates = 1 / torch.pow(10000, (2 * torch.arange(d_model) // 2).float() / d_model)
    positions = torch.arange(position).unsqueeze(1)
    angle_rads = positions * angle_rates.unsqueeze(0)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = torch.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = torch.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads.unsqueeze(0)  # (1, position, d_model)
    return pos_encoding


def scaled_dot_product_attention(q, k, v, mask=None):
    matmul_qk = torch.matmul(q, k.transpose(-2, -1))
    dk = q.size()[-1]
    scaled_attention_logits = matmul_qk / math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits = scaled_attention_logits.masked_fill(mask == 0, -1e9)

    attention_weights = torch.nn.functional.softmax(scaled_attention_logits, dim=-1)
    output = torch.matmul(attention_weights, v)
    return output, attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.depth = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.transpose(1, 2)  # (batch_size, num_heads, seq_len, depth)

    def forward(self, v, k, q, mask=None):
        batch_size = q.size(0)

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attn_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = scaled_attention.transpose(1, 2).contiguous()
        concat_attention = scaled_attention.view(batch_size, -1, self.num_heads * self.depth)

        output = self.dense(concat_attention)
        return output, attn_weights


def point_wise_feed_forward_network(d_model, dff):
    return nn.Sequential(
        nn.Linear(d_model, dff),
        nn.ReLU(),
        nn.Linear(dff, d_model)
    )


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.5):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x, mask=None):
        attn_output, _ = self.mha(x, x, x, mask)
        out1 = self.layernorm1(x + self.dropout1(attn_output))
        ffn_output = self.ffn(out1)
        out2 = self.layernorm2(out1 + self.dropout2(ffn_output))
        return out2

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.5, max_len=1000):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.output_dim = d_model

        pos_encoding = positional_encoding(max_len, d_model)
        self.register_buffer('pos_encoding', pos_encoding)

        self.enc_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, dff, rate)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(rate)

    def forward(self, x, mask=None):
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :].to(x.dtype)
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask)

        # return x
        # return x[:, 64, :] # 129 序列长度，获取突变位点作为整个表示
        center_idx = (seq_len - 1) // 2
        return x[:, center_idx, :]