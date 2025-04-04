import torch
import torch.nn as nn
import torch.optim as optim
import gc
from models.protein_sequence_classification import ProteinSequenceClassification

class MAMLFewShotClassifier(nn.Module):
    def __init__(self, pre_trained_model: ProteinSequenceClassification, inner_lr=0.01, inner_steps=5):
        super(MAMLFewShotClassifier, self).__init__()
        self.pre_trained_model = pre_trained_model  # No DataParallel here as we need parameter access
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        
    def forward(self, support_sequences, support_labels, query_sequences, batch_size):
        # Clone the model for adaptation
        adapted_model = self._clone_model(self.pre_trained_model)
        
        # Adapt the model on support set
        adapted_model = self._adapt(adapted_model, support_sequences, support_labels)
        
        # Evaluate on query set
        n_sequences = len(query_sequences)
        outputs = []
        
        for i in range(0, n_sequences, batch_size):
            mini_batch = query_sequences[i: i + batch_size]
            with torch.no_grad():
                logits = adapted_model(mini_batch)
            outputs.append(logits)
            
            # Cleanup
            del mini_batch
            torch.cuda.empty_cache()
            
        return torch.cat(outputs)
    
    def _clone_model(self, model):
        # Create a copy of the model with the same parameters
        clone = type(model)(*model.__init_args__, **model.__init_kwargs__)
        clone.load_state_dict(model.state_dict())
        return clone.to(model.device)
    
    def _adapt(self, model, support_sequences, support_labels):
        # Perform inner loop adaptation
        criterion = nn.CrossEntropyLoss()
        
        for _ in range(self.inner_steps):
            logits = model(support_sequences)
            loss = criterion(logits, support_labels)
            
            # Manual parameter update
            grads = torch.autograd.grad(loss, model.parameters())
            for p, g in zip(model.parameters(), grads):
                p.data.sub_(self.inner_lr * g)
                
        return model 