import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
from models.protein_sequence_classification import ProteinSequenceClassification

class SiameseNetworkFewShotClassifier(nn.Module):
    def __init__(self, pre_trained_model: ProteinSequenceClassification):
        super(SiameseNetworkFewShotClassifier, self).__init__()
        self.pre_trained_model = nn.DataParallel(pre_trained_model)
        
    def forward(self, support_sequences, support_labels, query_sequences, batch_size):
        # Get embeddings for support set
        support_features, _ = self.pre_trained_model(support_sequences, embedding_only=True)
        
        # Process query sequences in batches
        n_sequences = len(query_sequences)
        outputs = []
        
        for i in range(0, n_sequences, batch_size):
            mini_batch = query_sequences[i: i + batch_size]
            query_features, _ = self.pre_trained_model(mini_batch, embedding_only=True)
            
            # Calculate L1 distance between query and all support samples
            batch_size = query_features.shape[0]
            n_support = support_features.shape[0]
            
            # Reshape for broadcasting
            q = query_features.unsqueeze(1).repeat(1, n_support, 1)  # [batch_size, n_support, embedding_dim]
            s = support_features.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size, n_support, embedding_dim]
            
            # Calculate L1 distance
            distances = torch.abs(q - s).sum(dim=2)  # [batch_size, n_support]
            
            # For each query, find the nearest support sample and use its label
            min_indices = torch.argmin(distances, dim=1)  # [batch_size]
            predicted_labels = support_labels[min_indices]
            
            # Convert to one-hot
            n_classes = len(torch.unique(support_labels))
            one_hot = F.one_hot(predicted_labels, n_classes).float()
            outputs.append(one_hot)
            
            # Cleanup
            del mini_batch, query_features, distances
            torch.cuda.empty_cache()
            
        return torch.cat(outputs) 