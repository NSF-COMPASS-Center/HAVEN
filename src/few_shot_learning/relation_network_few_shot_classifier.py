import torch
import torch.nn as nn
import gc
from models.protein_sequence_classification import ProteinSequenceClassification

class RelationModule(nn.Module):
    def __init__(self, input_size):
        super(RelationModule, self).__init__()
        self.relation_net = nn.Sequential(
            nn.Linear(input_size * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        return self.relation_net(x)

class RelationNetworkFewShotClassifier(nn.Module):
    def __init__(self, pre_trained_model: ProteinSequenceClassification):
        super(RelationNetworkFewShotClassifier, self).__init__()
        self.pre_trained_model = nn.DataParallel(pre_trained_model)
        self.embedding_dim = pre_trained_model.embedding_dim  # Assuming this attribute exists
        self.relation_module = RelationModule(self.embedding_dim)
        
    def forward(self, support_sequences, support_labels, query_sequences, batch_size):
        # Compute class prototypes
        prototypes = []
        for label in torch.unique(support_labels):
            label_support_features, _ = self.pre_trained_model(
                torch.index_select(support_sequences, dim=0,
                                  index=torch.nonzero(support_labels == label).squeeze()),
                embedding_only=True
            )
            prototypes.append(label_support_features.mean(0))
            del label_support_features
            torch.cuda.empty_cache()
            
        prototypes = torch.stack(prototypes)  # n_way x embedding_dim
        n_way = prototypes.shape[0]
        
        # Process query sequences in batches
        n_sequences = len(query_sequences)
        outputs = []
        
        for i in range(0, n_sequences, batch_size):
            mini_batch = query_sequences[i: i + batch_size]
            query_features, _ = self.pre_trained_model(mini_batch, embedding_only=True)
            
            # For each query, compare with each prototype
            batch_size = query_features.shape[0]
            relations = []
            
            for j in range(n_way):
                # Repeat prototype to match batch size
                prototype = prototypes[j].unsqueeze(0).repeat(batch_size, 1)
                
                # Concatenate query features with prototype
                pairs = torch.cat([query_features, prototype], dim=1)
                
                # Compute relation score
                relation_score = self.relation_module(pairs)
                relations.append(relation_score)
            
            # Stack relation scores for all classes
            relation_scores = torch.cat(relations, dim=1)
            outputs.append(relation_scores)
            
            # Cleanup
            del mini_batch, query_features, relations
            torch.cuda.empty_cache()
            
        return torch.cat(outputs) 