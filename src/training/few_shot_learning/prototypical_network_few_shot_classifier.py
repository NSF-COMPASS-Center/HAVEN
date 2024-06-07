import torch
import torch.nn as nn


class PrototypicalNetworkFewShotClassifier(nn.Module):
    def __init__(self, pre_trained_model):
        super(PrototypicalNetworkFewShotClassifier, self).__init__()
        self.pre_trained_model = pre_trained_model

    def forward(self, support_sequences, support_labels, query_sequences):
        support_features = self.pre_trained_model(support_sequences).embeddings

        # number of unique classes in the support set
        n_way = len(torch.unique(support_labels)) # nunique returns the labels in sorted order
        prototypes = torch.cat([
            # torch.nonzero gives the indices with non-zero elements
            # we assume the labels are always (0, 1, ...., n_way-1) hence we use range(n_way)
            support_features[torch.nonzero(support_labels == label)].mean(0) for label in range(n_way)
        ])

        query_features = self.pre_trained_model(query_sequences).embeddings
        # cdist will compute the l2-norm aka euclidean distance ||x1-x2||^2
        # we negate the distance: lesser the distance to a prototype, more likely to be the class label of the prototype
        self.output = -torch.cdist(query_features, prototypes) # shape n_query_features X n_way
        return self.output

