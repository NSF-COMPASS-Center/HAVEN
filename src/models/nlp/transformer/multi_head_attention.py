import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    query $Q \in \mathbb{R}^{n \times d_q}$, \\
    key $K \in \mathbb{R}^{n \times d_k}$, \\
    value  $V\in \mathbb{R}^{n \times d_v}$ \\

    where $n$: number of input tokens processed simultaneoulsy \\
    $\text{attention}(Q, K, V)$ = $\text{softmax} \left(\frac{QK^T}{\sqrt{d_k}}\right)$

    Assumption: $d_q = d_k$
    :param Q: Query $\in \mathbb{R}^{n \times d_q}$
    :param K: Key $K \in \mathbb{R}^{n \times d_k}$
    :param V: Value $V\in \mathbb{R}^{n \times d_v}$
    :param mask:
    :return:
    """
    d_k = K.size(-1)  # key vector dimension
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(
        d_k)  # transpose(-2, -1) is same as transpose(0, 1) for 2D tensor

    # mask is used while training the decoder
    if mask is not None:
        mask = mask.unsqueeze(1) # add a dimension for the heads in multihead attention
        # replace all zero entries with negative infinity
        scores = scores.masked_fill(mask, -1e9)

    # softmax
    attn = F.softmax(scores, dim=-1)

    return torch.matmul(attn, V), attn


class MultiHeadAttention(nn.Module):
    """
    number of heads = $h = 8$
    $d$: dimension of the embeddings in the model
    query $Q' \in \mathbb{R}^{n \times d}$,
    key $K' \in \mathbb{R}^{n \times d}$,
    value  $V'\in \mathbb{R}^{n \times d}$

    $\text{head}_i = \text{attention}(Q'W_i^Q, K'W_i^K, V'W_i^V)$
    $W_i^Q \in \mathbb{R}^{d \times d}, W_i^K \in \mathbb{R}^{d \times d}, W_i^V \in \mathbb{R}^{d \times d}$

    Assumptions
    $d_q = d_k = d_v = 64$
    $h=8$
    $d = hd_v=hd_k=hd_q = 8*64=512$ (implementing multi=head attention using a singlehead attention
    $\text{MultiHead} (Q, K, V) = \text{concat}(\text{head}_1, \text{head}_2, \dots, \text{head}_h)W^O$
    $W^O \in \mathbb{R}^{hd_v \times d}$
    """
    def __init__(self, h, d):
        super(MultiHeadAttention, self).__init__()

        self.h = h
        # d_v = d_k = d_q = d_attn_head (dimension for a single head of attention)
        self.d_attn_head = d // h

        self.W_Q = nn.Linear(d, d)
        self.W_K = nn.Linear(d, d)
        self.W_V = nn.Linear(d, d)
        self.W_O = nn.Linear(d, d)
        self.self_attn = None

    def forward(self, Q_, K_, V_, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)  # same mask is applied to all h heads
        batch_size = Q_.size(0)

        # 1. Apply linear transformation (projection) in batches from
        # reshape to implement multihead attention using single head attention
        # where n = Key.size(1)  b x n x d
        # .view => b x n x h x d_attn_head
        # .transpose(1, 2) => b x h x n x d_attn_head
        Q = self.W_Q(Q_).view(batch_size, -1, self.h, self.d_attn_head).transpose(1, 2)
        K = self.W_K(K_).view(batch_size, -1, self.h, self.d_attn_head).transpose(1, 2)
        V = self.W_V(V_).view(batch_size, -1, self.h, self.d_attn_head).transpose(1, 2)

        # 2. Apply attention to all the projected vectors in batch
        X, self.self_attn = scaled_dot_product_attention(Q, K, V, mask=mask)

        # 3. Concat all the heads
        X = X.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_attn_head)

        del Q
        del K
        del V
        # 4. Apply final output linear transformation
        return self.W_O(X)