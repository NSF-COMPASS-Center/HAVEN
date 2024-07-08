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
        self.input_dim = input_dim
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
        # X: b x n where n is the maximum sequence length in the batch
        # 1. split into chunks
        batch_size = X.shape[0] # batch_size
        # split each sequence into smaller chunks along the dimension of the sequence length
        # let # chunks = n_chunks (depending on chunk_len and stride)
        X = X.unfold(dimension=1, size=self.chunk_len, step=self.stride) # b x n_c x chunk_len

        # reshape the tensor to individual sequences of chunk_len diregarding the sequence dimension (i.e. batch)
        # since we only need to generate embeddings for each chunk where the sequence identity does not matter
        # contiguous ensures contiguous memory allocation for every value in the tensor
        # this will enable reshaping using view which only changes the shape(view) of the tensor without creating a copy
        # reshape() 'might' create a copy. Hence, we use view() to save memory
        X = X.contiguous().view(-1, self.chunk_len) # (b * n_c) x chunk_len

        # generate embeddings
        X = self.pre_trained_model(X, mask=None)   # (b * n_c) x chunk_len x input_dim

        # reshape back into sequences of chunks, i.e. re-introduce the batch dimension.
        # here -1 will account for n_c which changes with the sequence length in every batch
        X = X.view(batch_size, -1, self.chunk_len, self.input_dim) # b x n_c x chunk_len x input_dim

        # mean of the embeddings of tokens in every chunk to create a representative vector for each chunk
        # mean along chunk_len dimension, i.e dim=2
        X = X.mean(dim=2) # b x n_c x input_dim

        # attention between the chunks in every sequence
        X = self.self_attn(X, X, X) # b x n_c x input_dim

        # feed-forward for projection + non-linear activation
        X = self.feed_forward(X) # b x n_c x input_dim

        # pool the embeddings of all chunks in the input sequence using mean to generate a vector embedding for each sequence
        X = X.mean(dim=1) # b x input_dim

        # input linear layer
        X = F.relu(self.linear_ip(X))
        # hidden
        self.embedding = X
        for linear_layer in self.linear_hidden_n:
            X = F.relu(linear_layer(X))
        y = self.linear_op(X)
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