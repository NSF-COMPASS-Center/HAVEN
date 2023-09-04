import torch.nn as nn
from utils import nn_utils
from utils.nlp_utils.embedding import EmbeddingLayer
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerModel(nn.Module):
    def __init__(self, n_tokens, max_seq_len, n_classes, N=6, input_dim=512, hidden_dim=1024, h=8):
        super(TransformerModel, self).__init__()
        self.embedding = EmbeddingLayer(vocab_size=n_tokens, max_seq_len=max_seq_len, dim=input_dim)

        encoder_layers = TransformerEncoderLayer(d_model=input_dim, nhead=h, dim_feedforward=hidden_dim, dropout=0.0,
                                                 batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, N, mask_check=False)

        self.linear = nn.Linear(input_dim, n_classes)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, X):
        X = self.embedding(X)
        X = self.transformer_encoder(X, mask=None)
        # pool the embeddings of all tokens using mean
        X = X.mean(dim=1)

        y = self.linear(X)
        return y


def get_transformer_model(model):
    tf_model = TransformerModel(n_tokens=model["n_tokens"],
                                max_seq_len=model["max_seq_len"],
                                n_classes=model["n_classes"],
                                N=model["depth"],
                                input_dim=model["input_dim"],
                                hidden_dim=model["hidden_dim"],
                                h=model["n_heads"])
    print(tf_model)
    print("Number of parameters = ", sum(p.numel() for p in tf_model.parameters() if p.requires_grad))
    return tf_model.to(nn_utils.get_device())
