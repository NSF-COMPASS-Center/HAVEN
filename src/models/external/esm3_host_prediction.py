from utils import nn_utils
from models.protein_sequence_classification import ProteinSequenceClassification
from huggingface_hub import login
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig
import torch

class ESM3_VirusHostPrediction(ProteinSequenceClassification):
    """
     Using ESM3 (https://github.com/evolutionaryscale/esm) for Virus Host Prediction
    """
    def __init__(self, input_dim, hidden_dim, n_mlp_layers, n_classes, max_seq_length, model_name):
        super(ESM3_VirusHostPrediction, self).__init__(input_dim, hidden_dim, n_mlp_layers, n_classes,
                                                            batch_norm=True)

        self.pre_trained_model = self.initialize_pre_trained_model(model_name)
        self.pre_trained_model.eval()

        # Login to Hugging Face Hub (API key needed)
        # will prompt to get an API key and accept the ESM3 license.
        login()

    def initialize_pre_trained_model(self, model_name):
        pre_trained_model = ESM3.from_pretrained(model_name)
        return pre_trained_model.to(nn_utils.get_device())

    def get_embedding(self, X):
        sequence_embeddings = []
        for sequence in X:
            seq_length = len(sequence)
            protein = ESMProtein(sequence=sequence)
            protein_tensor = self.pre_trained_model.encode(protein)

            output = self.pre_trained_model.forward_and_sample(
                protein_tensor, SamplingConfig(return_per_residue_embeddings=True)
            )

            # get per-sequence embedding via averaging (excluding start, and end tokens)
            sequence_embedding = output.per_residue_embedding[1: seq_length + 1].mean(dim=0)
            sequence_embeddings.append(sequence_embedding)

        return torch.stack(sequence_embeddings)

    def get_model(model_params) -> ProteinSequenceClassification:
        model = ESM3_VirusHostPrediction(input_dim=model_params["input_dim"],
                                         hidden_dim=model_params["hidden_dim"],
                                         n_mlp_layers=model_params["n_mlp_layers"],
                                         n_classes=model_params["n_classes"],
                                         max_seq_length=model_params["max_seq_length"],
                                         model_name=model_params["fine_tuned_model_name"])
        print(model)
        print("ESM3_VirusHostPrediction: Number of parameters = ",
              sum(p.numel() for p in model.parameters() if p.requires_grad))
        return ProteinSequenceClassification.return_model(model, model_params["data_parallel"])