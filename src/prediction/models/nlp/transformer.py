import torch.nn as nn
from prediction.models.nlp.embedding import EmbeddingLayer, ConvolutionEmbeddingLayer
from prediction.models.nlp.encoder import EncoderLayer, Encoder


class ClassificationTransformer(nn.Module):
    def __init__(self, n_tokens, max_seq_len, n_classes, N=6, d=512, d_ff=2048, h=8):
        super(ClassificationTransformer, self).__init__()
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


class ClassificationTransformer_Conv1D(nn.Module):
    def __init__(self, n_tokens, kernel_size, stride, padding, max_seq_len, n_classes, N=6, d=512, d_ff=2048, h=8):
        super(ClassificationTransformer_Conv1D, self).__init__()

        self.embedding = ConvolutionEmbeddingLayer(n_tokens, max_seq_len, d, kernel_size, stride, padding)
        self.encoder = Encoder(EncoderLayer(h, d, d_ff), N)
        self.linear = nn.Linear(d, n_classes)

    def forward(self, X):
        X = self.embedding(X)
        X = self.encoder(X)
        # pool the embeddings of all tokens using mean
        X = X.mean(dim=1)

        y = self.linear(X)
        return y


def get_transformer_model(model):
    if model["with_convolution"]:
        model = ClassificationTransformer_Conv1D(n_tokens=model["n_tokens"],
                                                 kernel_size=model["kernel_size"],
                                                 stride=model["stride"],
                                                 padding=model["padding"],
                                                 max_seq_len=model["max_sequence_length"],
                                                 n_classes=model["n_classes"],
                                                 N=model["depth"],
                                                 d=model["dim"],
                                                 d_ff=2 * model["dim"],
                                                 h=model["n_heads"])
    else:
        model = ClassificationTransformer(n_tokens=model["n_tokens"],
                                          max_seq_len=model["max_sequence_length"],
                                          n_classes=model["n_classes"],
                                          N=model["depth"],
                                          d=model["dim"],
                                          d_ff=2048,
                                          h=model["n_heads"])
    print(model)
    print("Number of parameters = ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model
