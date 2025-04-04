import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
from models.protein_sequence_classification import ProteinSequenceClassification

class GNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.W = nn.Linear(in_features, out_features)
        
    def forward(self, x, adj):
        # x: node features [n_nodes, in_features]
        # adj: adjacency matrix [n_nodes, n_nodes]
        h = torch.matmul(adj, x)  # Message passing
        return self.W(h)  # Transform

class GNNFewShotClassifier(nn.Module):
    def __init__(self, pre_trained_model: ProteinSequenceClassification):
        super(GNNFewShotClassifier, self).__init__()
        self.pre_trained_model = nn.DataParallel(pre_trained_model)
        self.embedding_dim = pre_trained_model.embedding_dim  # Assuming this attribute exists
        
        # GNN layers
        self.gnn1 = GNNLayer(self.embedding_dim, 256)
        self.gnn2 = GNNLayer(256, 128)
        self.classifier = nn.Linear(128, 1)
        
    def forward(self, support_sequences, support_labels, query_sequences, batch_size):
        # Get embeddings for support set
        support_features, _ = self.pre_trained_model(support_sequences, embedding_only=True)
        
        # Process query sequences in batches
        n_sequences = len(query_sequences)
        n_classes = len(torch.unique(support_labels))
        outputs = []
        
        for i in range(0, n_sequences, batch_size):
            mini_batch = query_sequences[i: i + batch_size]
            query_features, _ = self.pre_trained_model(mini_batch, embedding_only=True)
            
            batch_outputs = []
            for query in query_features:
                # For each query, create a graph with support samples
                # The query is connected to all support samples
                
                # Combine query and support features
                query_expanded = query.unsqueeze(0)
                all_features = torch.cat([query_expanded, support_features], dim=0)
                
                # Create adjacency matrix - query connects to all support samples
                n_nodes = all_features.shape[0]
                adj = torch.zeros(n_nodes, n_nodes, device=query.device)
                
                # Query (index 0) connects to all support samples
                adj[0, 1:] = 1
                adj[1:, 0] = 1
                
                # Support samples of same class connect to each other
                for c in range(n_classes):
                    class_indices = torch.nonzero(support_labels == c).squeeze() + 1  # +1 because query is at index 0
                    for idx1 in class_indices:
                        for idx2 in class_indices:
                            adj[idx1, idx2] = 1
                
                # Normalize adjacency matrix
                degree = adj.sum(1)
                degree_inv_sqrt = torch.pow(degree + 1e-6, -0.5)
                adj_normalized = degree_inv_sqrt.unsqueeze(1) * adj * degree_inv_sqrt.unsqueeze(0)
                
                # Apply GNN layers
                h = F.relu(self.gnn1(all_features, adj_normalized))
                h = F.relu(self.gnn2(h, adj_normalized))
                
                # Classify query node
                query_embedding = h[0]
                
                # Calculate similarity to each class prototype
                class_scores = []
                for c in range(n_classes):
                    class_indices = torch.nonzero(support_labels == c).squeeze() + 1
                    class_embeddings = h[class_indices]
                    prototype = class_embeddings.mean(0)
                    similarity = F.cosine_similarity(query_embedding.unsqueeze(0), prototype.unsqueeze(0))
                    class_scores.append(similarity)
                
                class_scores = torch.stack(class_scores)
                batch_outputs.append(class_scores)
            
            outputs.append(torch.stack(batch_outputs))
            
            # Cleanup
            del mini_batch, query_features
            torch.cuda.empty_cache()
            
        return torch.cat(outputs) 