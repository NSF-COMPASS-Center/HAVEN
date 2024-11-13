from utils import nn_utils
from models.protein_sequence_classification import ProteinSequenceClassification
import esm
import torch
import os

class ESM2_VirusHostPrediction(ProteinSequenceClassification):
    """
     Using ESM2 (https://github.com/facebookresearch/esm) for Virus Host Prediction
    """
    def __init__(self, input_dim, hidden_dim, n_mlp_layers, n_classes, max_seq_length, model_name, repr_layer):
        super(ESM2_VirusHostPrediction, self).__init__(input_dim, hidden_dim, n_mlp_layers, n_classes,
                                                            batch_norm=True)
        self.repr_layer = repr_layer
        self.alphabet, self.tokenizer, self.pre_trained_model = self.initialize_pre_trained_model(model_name)
        self.pre_trained_model.eval()

    def initialize_pre_trained_model(self, model_name):
        pre_trained_model, alphabet = getattr(esm.pretrained, model_name)()

        tokenizer = alphabet.get_batch_converter()
        return alphabet, tokenizer, pre_trained_model.to(nn_utils.get_device())


    def get_embedding(self, X):
        # tokenization
        _, _, batch_tokens = self.tokenizer(X)
        batch_tokens = batch_tokens.to(nn_utils.get_device())

        batch_token_lengths = (batch_tokens != self.alphabet.padding_idx).sum(1) # equal to lengths of sequences + 2

        # get pre-residue embedding
        with torch.no_grad():
            output = self.pre_trained_model(batch_tokens, repr_layers=[self.repr_layer])
        token_embeddings = output["representations"][self.repr_layer]

        # get per-sequence embedding via averaging (excluding padding, start, and end tokens)
        sequence_embeddings = []
        for i, token_length in enumerate(batch_token_lengths):
            sequence_embedding = token_embeddings[i, 1: token_length - 1].mean(0)
            sequence_embeddings.append(sequence_embedding)

        return torch.stack(sequence_embeddings)


    def get_model(model_params) -> ProteinSequenceClassification:
        model = ESM2_VirusHostPrediction(input_dim=model_params["input_dim"],
                                         hidden_dim=model_params["hidden_dim"],
                                         n_mlp_layers=model_params["n_mlp_layers"],
                                         n_classes=model_params["n_classes"],
                                         max_seq_length=model_params["max_seq_length"],
                                         model_name=model_params["fine_tuned_model_name"],
                                         repr_layer=model_params["repr_layer"])
        print(model)
        print("ESM2_VirusHostPrediction: Number of parameters = ",
              sum(p.numel() for p in model.parameters() if p.requires_grad))
        return ProteinSequenceClassification.return_model(model, model_params["data_parallel"])