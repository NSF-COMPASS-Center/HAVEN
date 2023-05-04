import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN_2L(torch.nn.Module):
    def __init__(self, n_node_features, h, n_output_features):
        super(GCN_2L, self).__init__()
        self.gcn_l1 = GCNConv(n_node_features, h)
        self.gcn_l2 = GCNConv(h, n_output_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gcn_l1(x, edge_index)
        x = F.relu(x)
        x = self.gcn_l2(x, edge_index)
        x = F.relu(x)
        return x
