import torch
import torch.nn as nn


class PrototypicalNetworkFewShotClassifier(nn.Module):
    def __init__(self, pre_trained_model):
        super(PrototypicalNetworkFewShotClassifier, self).__init__()
        self.pre_trained_model = pre_trained_model

    def forward(self, support_sequences, support_labels, query_sequences):
        support_features = self.pre_trained_model.get_embedding(support_sequences)

        # number of unique classes in the support set
        n_way = len(torch.unique(support_labels)) # nunique returns the labels in sorted order
        prototypes = torch.cat([
            # torch.nonzero gives the indices with non-zero elements
            # we assume the labels are always (0, 1, ...., n_way-1) hence we use range(n_way)
            support_features[torch.nonzero(support_labels == label)].mean(0) for label in range(n_way)
        ])

        query_features = self.pre_trained_model.get_embedding(query_sequences)
        # cdist will compute the l2-norm aka euclidean distance ||x1-x2||^2
        # we negate the distance: lesser the distance to a prototype, more likely to be the class label of the prototype
        self.output = -torch.cdist(query_features, prototypes) # shape n_query_features X n_way
        return self.output


    def predict(self, support_sequences, support_labels, query_sequences, batch_size):
        support_features = self.pre_trained_model.get_embedding(support_sequences)

        # number of unique classes in the support set
        n_way = len(torch.unique(support_labels)) # nunique returns the labels in sorted order
        prototypes = torch.cat([
            # torch.nonzero gives the indices with non-zero elements
            # we assume the labels are always (0, 1, ...., n_way-1) hence we use range(n_way)
            support_features[torch.nonzero(support_labels == label)].mean(0) for label in range(n_way)
        ])
        n_query_sequences = len(query_sequences)

        # compute queries in batches of fixed size as per the memory constraints of the server
        query_features = []

        for i in range(0, n_query_sequences, batch_size):
            query_features.append(self.pre_trained_model.get_embedding(query_sequences[i : i + batch_size]))

        query_features = torch.cat(query_features)
        # cdist will compute the l2-norm aka euclidean distance ||x1-x2||^2
        # we negate the distance: lesser the distance to a prototype, more likely to be the class label of the prototype
        self.output = -torch.cdist(query_features, prototypes) # shape n_query_features X n_way
        return self.output
