import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from prediction.models.gnn.gcn import GCN_2L


class GNN_Pipeline(torch.nn.ModuleList):
    def __init__(self, n_node_features, gnn_h, n_gnn_output_features, mlp_h, n_classes):
        super(GNN_Pipeline, self).__init__()
        self.host_prot_gcn = GCN_2L(n_node_features=n_node_features,
                                    h=gnn_h,
                                    n_output_features=n_gnn_output_features)
        self.virus_prot_gcn = GCN_2L(n_node_features=n_node_features,
                                    h=gnn_h,
                                    n_output_features=n_gnn_output_features)
        self.linear_1 = nn.Linear(2 * n_gnn_output_features, mlp_h)
        self.linear_2 = nn.Linear(mlp_h, n_classes)

    def forward(self, data_host_prot, data_virus_prot):
        # Node embeddings from graph convolution
        x_host_prot = self.host_prot_gcn(data_host_prot)
        x_virus_prot = self.receptor_gcn(data_virus_prot)

        # Readout layer: graph embedding = average of all node embeddings
        x_host_prot = global_mean_pool(x_host_prot)  # b X n_gnn_output_features
        x_virus_prot = global_mean_pool(x_virus_prot)  # b X n_gnn_output_features

        # MLP layer for classification
        x_prot_interaction = torch.concat((x_host_prot, x_virus_prot), dim=1)  # b X 2*n_gnn_output_features
        x_prot_interaction = F.relu(self.linear_1(x_prot_interaction))  # b X mlp_h

        return self.linear_2(x_prot_interaction)  # b X n_classes


def get_gnn_model(model):
    gnn_model = GNN_Pipeline(n_node_features=model["n_node_features"],
                             gnn_h=model["gnn_h"],
                             n_gnn_output_features=model["n_gnn_output_features"],
                             mlp_h=model["mlp_h"],
                             n_classes=model["n_classes"])
    print(gnn_model)
    print("Number of parameters = ", sum(p.numel() for p in gnn_model.parameters() if p.requires_grad))
    return gnn_model
