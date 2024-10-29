from torch import nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from utils import nn_utils
from models.protein_sequence_classification import ProteinSequenceClassification

class BERT_VirusHostPrediction(ProteinSequenceClassification):
    """
    Fine-tune a vanilla pre-trained BERT model using the whole sequence as input.
    """
    def __init__(self, pre_trained_model, cls_token, input_dim, hidden_dim, n_mlp_layers, n_classes):
        super(ProteinSequenceClassification, self).__init__(input_dim, hidden_dim, n_mlp_layers, n_classes, batch_norm=True)
        self.pre_trained_model = pre_trained_model
        self.cls_token = cls_token


    def get_embedding(self, X):
        X = self.pre_trained_model(X, mask=None)

        if self.cls_token:
            # OPTION 1: representative vector for each sequence = CLS token embedding in every segment
            X = X[:, 0, :]
        else:
            # pool the pre_trained_model embeddings of all tokens in the input sequence using mean
            X = X.mean(dim=1)
        return X

    # def forward() : use the template implementation in ProteinSequenceClassification


    def get_host_prediction_model(model_params, data_parallel) -> BERT_VirusHostPrediction:
        model = BERT_VirusHostPrediction(pre_trained_model=model_params["pre_trained_model"],
                                         input_dim=model_params["input_dim"],
                                         cls_token=model_params["cls_token"],
                                         hidden_dim=model_params["hidden_dim"],
                                         n_mlp_layers=model_params["n_mlp_layers"],
                                         n_classes=model_params["n_classes"])
        print(model)
        print("BERT_VirusHostPrediction: Number of parameters = ", sum(p.numel() for p in host_prediction_model.parameters() if p.requires_grad))
        return VirusHostPredictionBase.return_model(model, data_parallel)
