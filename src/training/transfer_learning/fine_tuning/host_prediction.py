from torch import nn
import torch.nn.functional as F

from utils import nn_utils


class HostPrediction(nn.Module):
    def __init__(self, pre_trained_model, input_dim, hidden_dim, depth, n_classes):
        super(HostPrediction, self).__init__()
        self.pre_trained_model = pre_trained_model
        # first linear layer: input_dim --> hidden_dim
        self.linear_ip = nn.Linear(input_dim, hidden_dim)

        self.linear_hidden = nn.Linear(hidden_dim, hidden_dim)
        # intermediate hidden layers (number = N): hidden_dim --> hidden_dim
        self.linear_hidden_n = nn_utils.create_clones(self.linear_hidden, depth)

        # last linear layer: hidden_dim--> n_classes
        self.linear_op = nn.Linear(hidden_dim, n_classes)

    def get_embedding(self, X):
        X = self.pre_trained_model(X, mask=None)
        # pool the pre_trained_model embeddings of all tokens in the input sequence using mean
        X = X.mean(dim=1)
        # input linear layer
        X = F.relu(self.linear_ip(X))
        # hidden
        for linear_layer in self.linear_hidden_n:
            X = F.relu(linear_layer(X))
        return X

    def forward(self, X):
        # embedding to be used for interpretability of the fine-tuned model
        self.fine_tuned_embedding = self.get_embedding(X)

        y = self.linear_op(X)
        return y


def get_host_prediction_model(task):
    host_prediction_model = HostPrediction(pre_trained_model=task["pre_trained_model"],
                                           input_dim=task["input_dim"],
                                           hidden_dim=task["hidden_dim"],
                                           depth=task["depth"],
                                           n_classes=task["n_classes"])
    print(host_prediction_model)
    print("Number of parameters = ", sum(p.numel() for p in host_prediction_model.parameters() if p.requires_grad))
    return host_prediction_model.to(nn_utils.get_device())
