import torch
import torch.nn as nn
import gc

class PrototypicalNetworkFewShotClassifier(nn.Module):
    def __init__(self, pre_trained_model):
        super(PrototypicalNetworkFewShotClassifier, self).__init__()
        self.pre_trained_model = pre_trained_model

    def forward(self, support_sequences, support_labels, query_sequences, batch_size):
        # compute prototypes for each label
        prototypes = []

        # unique returns the labels in sorted order
        # we assume the labels are always (0, 1, ...., n_way-1)
        for label in torch.unique(support_labels):
            # assuming n_shot is within the server memory constraints
            # i.e, n_shot <= batch_size
            label_support_features = self.pre_trained_model.get_embedding(
                support_sequences[
                    torch.nonzero(support_labels == label).squeeze() # torch.nonzero gives the indices with non-zero elements but it adds a dimension as [n, 1] hence we use squeeze to remove the added extra dimension
                ]
            )
            # prototype is the mean of the support features
            prototypes.append(label_support_features.mean(0))
            del label_support_features # mark for deletion

        del support_sequences # mark for deletion
        gc.collect() # garbage collection to free up memory
        
        # assuming order is maintained and the prototype vector for each label is located at the corresponding index
        prototypes = torch.stack(prototypes) # n_way X embedding_dimension

        # compute queries in batches of fixed size as per the memory constraints of the server
        query_features = self.get_embedding(query_sequences, batch_size)

        # cdist will compute the l2-norm aka euclidean distance ||x1-x2||^2
        # we negate the distance: lesser the distance to a prototype, more likely to be the class label of the prototype
        self.output = -torch.cdist(query_features, prototypes) # shape n_query_features X n_way
        return self.output

    # method to get generate embeddings for sequences in batches
    def get_embedding(self, sequences, batch_size):
        n_sequences = len(sequences)
        features = []

        for i in range(0, n_sequences, batch_size):
            features.append(self.pre_trained_model.get_embedding(sequences[i: i + batch_size]))

        return torch.cat(features)