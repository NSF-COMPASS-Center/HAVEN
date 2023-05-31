import torch.nn as nn
from prediction.models.nlp.embedding import EmbeddingLayer, ConvolutionEmbeddingLayer
from prediction.models.nlp.encoder import EncoderLayer, Encoder
from utils import nn_utils


class Transformer(nn.Module):
    def __init__(self, n_tokens, max_seq_len, n_classes, N=6, d=512, d_ff=1024, h=8):
        super(Transformer, self).__init__()
        self.embedding = EmbeddingLayer(vocab_size=n_tokens, max_seq_len=max_seq_len, dim=d)
        self.encoder = Encoder(EncoderLayer(h, d, d_ff), N)
        self.linear = nn.Linear(d, n_classes)

    def forward(self, X):
        X = self.embedding(X)
        X = self.encoder(X)
        # pool the embeddings of all tokens using mean
        X = X.mean(dim=1)

        y = self.linear(X)
        return y


class Transformer_Conv1D(Transformer):
    def __init__(self, n_tokens, kernel_size, stride, padding, max_seq_len, n_classes, N=6, d=512, d_ff=1024, h=8):
        super(Transformer_Conv1D, self).__init__(n_tokens, max_seq_len, n_classes, N, d, d_ff, h)
        self.embedding = ConvolutionEmbeddingLayer(n_tokens, max_seq_len, d, kernel_size, stride, padding)


def get_transformer_model(model):
    if model["with_convolution"]:
        tf_model = Transformer_Conv1D(n_tokens=model["n_tokens"],
                                                 kernel_size=model["kernel_size"],
                                                 stride=model["stride"],
                                                 padding=model["padding"],
                                                 max_seq_len=model["max_seq_len"],
                                                 n_classes=model["n_classes"],
                                                 N=model["depth"],
                                                 d=model["dim"],
                                                 d_ff=2 * model["dim"],
                                                 h=model["n_heads"])
    else:
        tf_model = Transformer(n_tokens=model["n_tokens"],
                                          max_seq_len=model["max_seq_len"],
                                          n_classes=model["n_classes"],
                                          N=model["depth"],
                                          d=model["dim"],
                                          d_ff=2048,
                                          h=model["n_heads"])
    # initialize model
    nn_utils.init_weights(model=tf_model,
                          initialization_type=model["weight_initialization"],
                          bias_init_value=0)
    print(tf_model)
    print("Number of parameters = ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Model weights =")
    print(tf_model.weight)
    return tf_model
