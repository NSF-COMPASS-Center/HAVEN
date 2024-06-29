import torch.nn as nn
from models.nlp.transformer.encoder import EncoderLayer, Encoder
from utils import nn_utils


# only encoder
class TransformerFNN(nn.Module):
    def __init__(self, embedding_model, max_chunk_len, stride):
        super(HierarchicalTransformer, self).__init__()
        self.embedding_model = embedding_model

        self.stride = stride

    def forward(self, X):
        X = self.embedding(X)
        X = self.encoder(X, mask=None) # output
        return X




def get_model(model):

    tf_model = TransformerEncoder(n_tokens=model["n_tokens"],
                                      max_seq_len=model["max_seq_len"],
                                      N=model["depth"],
                                      input_dim=model["input_dim"],
                                      hidden_dim=model["hidden_dim"],
                                      h=model["n_heads"])
    print(tf_model)
    print("Number of parameters = ", sum(p.numel() for p in tf_model.parameters() if p.requires_grad))
    return tf_model.to(nn_utils.get_device())