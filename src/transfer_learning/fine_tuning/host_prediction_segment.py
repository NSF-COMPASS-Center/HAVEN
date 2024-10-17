from torch import nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from utils import nn_utils, constants
import torch


class HostPredictionSegment(nn.Module):
    def __init__(self, pre_trained_model, segment_len, cls_token, input_dim, hidden_dim, stride=1, depth=2, n_classes=1):
        super(HostPredictionSegment, self).__init__()
        self.pre_trained_model = pre_trained_model
        self.segment_len = segment_len
        self.cls_token = cls_token
        self.stride = stride
        self.input_dim = input_dim

        # first linear layer: input_dim --> hidden_dim
        self.linear_ip = nn.Linear(input_dim, hidden_dim)
        self.batch_norm_ip = BatchNorm1d(hidden_dim)
        self.linear_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.batch_norm_hidden = BatchNorm1d(hidden_dim)
        # intermediate hidden layers (number = N): hidden_dim --> hidden_dim
        self.linear_hidden_n = nn_utils.create_clones(self.linear_hidden, depth)
        self.batch_norm_hidden_n = nn_utils.create_clones(self.batch_norm_hidden, depth)
        # last linear layer: hidden_dim--> n_classes
        self.linear_op = nn.Linear(hidden_dim, n_classes)

    def get_embedding(self, X):
        # X: b x n where n is the maximum sequence length in the batch
        # 1. split into segments
        batch_size = X.shape[0]  # batch_size
        # split each sequence into smaller segments along the dimension of the sequence length
        # let # segment = n_s (depending on segment_len and stride)
        X = X.unfold(dimension=1, size=self.segment_len, step=self.stride)  # b x n_s x segment_len

        # reshape the tensor to individual sequences of segment_len diregarding the sequence dimension (i.e. batch)
        # since we only need to generate embeddings for each segment where the sequence identity does not matter
        # contiguous ensures contiguous memory allocation for every value in the tensor
        # this will enable reshaping using view which only changes the shape(view) of the tensor without creating a copy
        # reshape() 'might' create a copy. Hence, we use view() to save memory
        X = X.contiguous().view(-1, self.segment_len)  # (b * n_s) x segment_len

        if self.cls_token:
            # add cls token to the beginning to every segment
            cls_tokens = torch.full(size=(X.shape[0], 1), fill_value=constants.CLS_TOKEN_VAL,
                                    device=nn_utils.get_device())
            X = torch.cat([cls_tokens, X], dim=1)

        # generate embeddings
        X = self.pre_trained_model(X, mask=None)  # (b * n_s) x segment_len x input_dim

        # reshape back into sequences of chunks, i.e. re-introduce the batch dimension.
        # here -1 will account for n_s which changes with the sequence length in every batch
        # we use segment_len + 1 to account for the added CLS token

        if self.cls_token:
            X = X.view(batch_size, -1, self.segment_len + 1, self.input_dim)  # b x n_s x segment_len + 1 x input_dim
        else:
            X = X.view(batch_size, -1, self.segment_len, self.input_dim)  # b x n_s x segment_len + 1 x input_dim

        if self.cls_token:
            # OPTION 1: representative vector for each segment = CLS token embedding in every segment
            X = X[:, :, 0, :]
        else:
            # OPTION 2: representative vector for each segment = mean of the embeddings of tokens in every segment
            # mean along segment_len dimension, i.e dim=2
            X = X.mean(dim=2)  # b x n_s x input_dim

        # pool the embeddings of all segments in the input sequence using mean to generate a vector embedding for each sequence
        X = X.mean(dim=1)  # b x input_dim

        return X

    def forward(self, X, embedding_only=False):
        batch_size = X.shape[0]  # batch_size
        X = self.get_embedding(X)
        if embedding_only:
            return X

        # input linear layer
        X = F.relu(self.linear_ip(X))
        if batch_size > 1:  # batch_norm is applicable only when batch_size is > 1
            X = self.batch_norm_ip(X)

        # hidden
        for i, linear_layer in enumerate(self.linear_hidden_n):
            X = F.relu(linear_layer(X))
            if batch_size > 1:  # batch_norm is applicable only when batch_size is > 1
                X = self.batch_norm_hidden_n[i](X)

        y = self.linear_op(X)
        return y


def get_host_prediction_model(task):
    host_prediction_model = HostPredictionSegment(pre_trained_model=task["pre_trained_model"],
                                                  input_dim=task["input_dim"],
                                                  cls_token=task["cls_token"],
                                                  stride=task["stride"],
                                                  segment_len=task["segment_len"],
                                                  hidden_dim=task["hidden_dim"],
                                                  depth=task["depth"],
                                                  n_classes=task["n_classes"])
    print(host_prediction_model)
    print("Number of parameters = ", sum(p.numel() for p in host_prediction_model.parameters() if p.requires_grad))
    return host_prediction_model.to(nn_utils.get_device())
