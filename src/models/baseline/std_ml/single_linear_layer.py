import torch
import torch.nn as nn
import torch.optim as optim


# Linear Layer
class SingleLinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SingleLinearLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)