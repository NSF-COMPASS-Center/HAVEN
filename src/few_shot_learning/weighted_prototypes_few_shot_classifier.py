import torch
import torch.nn as nn
import gc
from models.protein_sequence_classification import ProteinSequenceClassification

class WeightedPrototypesFewShotClassifier(nn.Module):
    def __init__(self, pre_trained_model: ProteinSequenceClassification):
        super(WeightedPrototypesFewShotClassifier, self).__init__()
        self.pre_trained_model = nn.DataParallel(pre_trained_model)

    def forward(self, support_sequences, support_labels, query_sequences, batch_size):
        # Compute all support features
        support_features, _ = self.pre_trained_model(X=support_sequences, embedding_only=True)
        
        # Compute query features
        n_sequences = len(query_sequences)
        query_features_list = []
        for i in range(0, n_sequences, batch_size):
            mini_batch = query_sequences[i: i + batch_size]
            features, _ = self.pre_trained_model(X=mini_batch, embedding_only=True)
            query_features_list.append(features)
        query_features = torch.cat(query_features_list)
        
        # Compute weighted prototypes for each label using attention
        prototypes = []
        for label in torch.unique(support_labels):
            label_indices = torch.nonzero(support_labels == label).squeeze()
            label_support_features = support_features[label_indices]
            
            # Compute similarity between each query and each support example
            similarities = torch.mm(query_features, label_support_features.transpose(0, 1))
            attention = torch.softmax(similarities, dim=1)
            
            # Compute a weighted prototype for each query example
            weighted_prototypes = torch.matmul(attention, label_support_features)
            prototypes.append(weighted_prototypes)
        
        # Stack prototypes along dimension 1
        prototypes = torch.stack(prototypes, dim=1)  # n_query x n_way x embedding_dim
        
        # Compute distances
        query_expanded = query_features.unsqueeze(1)  # n_query x 1 x embedding_dim
        distances = -torch.sum((query_expanded - prototypes)**2, dim=2)  # n_query x n_way
        
        return distances

    def compute_query_features_with_repetition(self, mini_batch, n_gpus):
        # if number of samples in the mini_batch is less than the number of gpus available, dataparallel will error out due to insufficient samples to split among the multiple GPUs
        # work around (though hacky):
        # 1. create copies of the mini_batch such that every GPU will have the same mini_batch
        # 2. use only the features from the first GPU for the mini_batch and ignore the embeddings from the remaining GPUs

        # 1 indicates not changing the size in that dimension
        # i.e, number of times times for repitition = 1 (technically zero), so no repitition along the columns(second dimension)
        mini_batch = mini_batch.repeat(n_gpus, 1)
        query_features, _ = self.pre_trained_model(X=mini_batch, embedding_only=True)
        # return only the features for the mini_batch from the first GPU
        # add a batch dimension, i.e. dimension at axis=0 with value=1
        return query_features[0].unsqueeze(0) 