import torch.nn as nn
from models.nlp.transformer.multi_head_attention import MultiHeadAttention
from models.nlp.transformer.feed_forward_layer import FeedForwardLayer
from utils import nn_utils
import torch
import torch.nn.functional as F

# only encoder
class TransformerAttention(nn.Module):
    def __init__(self, pre_trained_model, chunk_len, h=8, input_dim=512, hidden_dim=2048, stride=1, depth=2, n_classes=1):
        super(TransformerAttention, self).__init__()
        self.pre_trained_model = pre_trained_model
        self.self_attn = MultiHeadAttention(h, input_dim)
        self.feed_forward = FeedForwardLayer(input_dim, hidden_dim)
        self.chunk_len = chunk_len
        self.stride = stride

        # Classification block
        # first linear layer: input_dim --> hidden_dim
        self.linear_ip = nn.Linear(input_dim, hidden_dim)

        self.linear_hidden = nn.Linear(hidden_dim, hidden_dim)
        # intermediate hidden layers (number = N): hidden_dim --> hidden_dim
        self.linear_hidden_n = nn_utils.create_clones(self.linear_hidden, depth)

        # last linear layer: hidden_dim--> n_classes
        self.linear_op = nn.Linear(hidden_dim, n_classes)

    def forward(self, X):
        # X: b \times n
        # 1. split into chunks
        batch_size = X.shape[0] # batch_size
        X_chunk_emb = []
        for i in range(batch_size):
            X_i = X[i].unfold(dimension=0, size=self.chunk_len, step=self.stride)
            X_i = self.pre_trained_model(X_i, mask=None)
            X_i = X_i.mean(dim=1)
            X_chunk_emb.append(X_i)
        X = torch.stack(X_chunk_emb)
        X = self.self_attn(X, X, X)
        X = self.feed_forward(X)

        # pool the pre_trained_model embeddings of all tokens in the input sequence using mean
        X = X.mean(dim=1)
        # input linear layer
        X = F.relu(self.linear_ip(X))
        # hidden
        for linear_layer in self.linear_hidden_n:
            X = F.relu(linear_layer(X))
        # embedding to be used for interpretability of the fine-tuned model
        self.fine_tuned_embedding = X
        y = self.linear_op(self.fine_tuned_embedding)
        return y


def get_model(model):
    model = TransformerAttention(pre_trained_model=model["pre_trained_model"],
                                 chunk_len=model["chunk_len"],
                                 h = model["n_heads"],
                                 input_dim=model["input_dim"],
                                 hidden_dim=model["hidden_dim"],
                                 depth=model["depth"],
                                 stride=model["stride"],
                                 n_classes=model["n_classes"])
    print(model)
    print("Number of parameters = ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model.to(nn_utils.get_device())