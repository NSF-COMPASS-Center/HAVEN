import torch
import torch.nn as nn
import torch.nn.functional as F
from models.protein_sequence_classification import ProteinSequenceClassification

class MatchingNetworkFewShotClassifier(nn.Module):
    def __init__(self, pre_trained_model: ProteinSequenceClassification):
        super(MatchingNetworkFewShotClassifier, self).__init__()
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
            
            # Calculate cosine similarity between query and support features
            similarities = self._cosine_similarity(query_features, support_features)
            
            # Apply softmax to get attention weights
            attention = F.softmax(similarities, dim=1)
            
            # Get one-hot encoded labels
            n_classes = len(torch.unique(support_labels))
            y_one_hot = F.one_hot(support_labels, n_classes).float()
            
            # Weighted sum of labels based on attention
            outputs.append(torch.matmul(attention, y_one_hot))
            
            # Cleanup
            del mini_batch, query_features, similarities, attention
            torch.cuda.empty_cache()
            
        return torch.cat(outputs)
    
    def _cosine_similarity(self, a, b):
        # Compute cosine similarity between each query and all support samples
        a_norm = F.normalize(a, p=2, dim=1)
        b_norm = F.normalize(b, p=2, dim=1)
        return torch.matmul(a_norm, b_norm.transpose(0, 1)) 