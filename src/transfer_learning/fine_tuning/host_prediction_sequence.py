from torch import nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from utils import nn_utils


class HostPredictionSequence(nn.Module):
    def __init__(self, pre_trained_model, cls_token, input_dim, hidden_dim, depth, n_classes):
        super(HostPredictionSequence, self).__init__()
        self.pre_trained_model = pre_trained_model
        self.cls_token = cls_token

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
        X = self.pre_trained_model(X, mask=None)

        if self.cls_token:
            # OPTION 1: representative vector for each sequence = CLS token embedding in every segment
            X = X[:, 0, :]
        else:
            # pool the pre_trained_model embeddings of all tokens in the input sequence using mean
            X = X.mean(dim=1)
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
    host_prediction_model = HostPredictionSequence(pre_trained_model=task["pre_trained_model"],
                                                   input_dim=task["input_dim"],
                                                   cls_token=task["cls_token"],
                                                   hidden_dim=task["hidden_dim"],
                                                   depth=task["depth"],
                                                   n_classes=task["n_classes"])
    print(host_prediction_model)
    print("Number of parameters = ", sum(p.numel() for p in host_prediction_model.parameters() if p.requires_grad))
    return host_prediction_model.to(nn_utils.get_device())
