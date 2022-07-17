import numpy as np
import torch
import torch.nn as nn

from lesson5.attn import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(self,
                 input_dim: int,
                 head_dim: int,
                 n_heads: int,
                 emb_dim: int,
                 ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        # self.w_ks = [nn.Linear(input_dim, head_dim, bias=False) for _ in range(n_heads)]
        self.w_key = nn.Linear(input_dim, head_dim * n_heads, bias=False)
        self.w_query = nn.Linear(input_dim, head_dim * n_heads, bias=False)
        self.w_value = nn.Linear(input_dim, head_dim * n_heads, bias=False)

        self.attn = ScaledDotProductAttention(temperature=np.power(head_dim, 0.5))

        self.proj = nn.Linear(n_heads * head_dim, emb_dim, bias=False)

    def forward(self, k, q, v):
        B, T, _ = k.size()
        k = self.w_key(k) # B x T x n_heads * head_dim
        q = self.w_query(q)
        v = self.w_value(v)

        # B x T x n_heads x head_dim
        k = k.view(B, T, self.n_heads, self.head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim)
        v = v.view(B, T, self.n_heads, self.head_dim)

        # n_heads x B x T x head_dim -> n_heads * B x T x head_dim
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, T, self.head_dim)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, T, self.head_dim)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, T, self.head_dim)

        output, attn = self.attn(q, k, v)

        output = output.view(self.n_heads, B, T, self.head_dim).permute(1, 2, 0, 3).contiguous().view(B, T, -1)

        output = self.proj(output)

        return output


class FFNet(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 dropout: float = 0.1,
                 ):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class FFTransformerBlock(nn.Module):
    def __init__(self,
                 input_dim: int,
                 head_dim: int,
                 n_heads: int,
                 emb_dim: int,
                 hidden_dim: int,
                 ):
        super().__init__()
        self.mutiheadattn = MultiHeadAttention(input_dim, head_dim, n_heads, emb_dim)
        self.ffnet = FFNet(input_dim, hidden_dim)

        self.ln1 = nn.LayerNorm()
        self.ln2 = nn.LayerNorm()

    def forward(self,
                x, # B x T x input_dim
                ):
        residiual = x
        x = self.mutiheadattn(x, x ,x)
        x = self.ln1(x + residiual)

        residiual = x
        x = self.ffnet(x)
        x = self.ln2(x + residiual)

        return x


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i)
                               for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    #change is None term on >= 0
    if padding_idx >= 0:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


class Encoder(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 emb_dim: int,
                 head_dim: int,
                 n_heads: int,
                 hidden_dim: int,
                 n_blocks: int,
                 ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(2000, emb_dim, padding_idx=0),
            freeze=True)

        self.layers = nn.ModuleList([
            FFTransformerBlock(emb_dim, head_dim, n_heads, emb_dim, hidden_dim) for _ in range(n_blocks)
        ])

    def forward(self,
                x, # B x T
                positions,
                ):
        x = self.emb(x) # B x T x emb_dim
        pos = self.position_enc(positions)
        x += pos

        for layer in self.layers:
            x = layer(x)

        return x
