import torch
import torch.nn as nn
from prediction.models.nlp.embedding import EmbeddingLayer, ConvolutionEmbeddingLayer
from prediction.models.nlp.encoder import EncoderLayer, Encoder


class Zoonoformer(nn.Module):
    def __init__(self, n_tokens, max_seq_len, n_classes, N=6, tf_d=512, d_ff=1024, h=8, gnn_d=512):
        super(Zoonoformer, self).__init__()
        self.embedding = EmbeddingLayer(vocab_size=n_tokens, max_seq_len=max_seq_len, dim=tf_d)
        self.encoder = Encoder(EncoderLayer(h, tf_d, d_ff), N)
        self.linear = nn.Linear(tf_d + gnn_d, n_classes)  # input=concatenated embeddings from tf and gnn

    def forward(self, X, X_gnn):
        X = self.embedding(X)
        X = self.encoder(X)
        # pool the embeddings of all tokens using mean
        X = X.mean(dim=1)

        # concat the embedding learned from tf with the embedding from the graphs
        X_tf_gnn = torch.cat(X, X_gnn, dim=1)
        y = self.linear(X_tf_gnn)
        return y


class Zoonoformer_Conv1D(Zoonoformer):
    def __init__(self, n_tokens, kernel_size, stride, padding, max_seq_len, n_classes, N=6, d=512, d_ff=1024, h=8):
        super(Zoonoformer_Conv1D, self).__init__(n_tokens, max_seq_len, n_classes, N, d, d_ff, h)
        self.embedding = ConvolutionEmbeddingLayer(n_tokens, max_seq_len, d, kernel_size, stride, padding)


def get_zoonoformer_model(model):
    if model["with_convolution"]:
        model = Zoonoformer_Conv1D(n_tokens=model["n_tokens"],
                                   kernel_size=model["kernel_size"],
                                   stride=model["stride"],
                                   padding=model["padding"],
                                   max_seq_len=model["max_seq_len"],
                                   n_classes=model["n_classes"],
                                   N=model["depth"],
                                   tf_d=model["tf_dim"],
                                   d_ff=2 * model["tf_dim"],
                                   h=model["n_heads"],
                                   gnn_d=model["n_gnn_output_features"])
    else:
        model = Zoonoformer(n_tokens=model["n_tokens"],
                            max_seq_len=model["max_seq_len"],
                            n_classes=model["n_classes"],
                            N=model["depth"],
                            d=model["tf_dim"],
                            d_ff=2048,
                            h=model["n_heads"],
                            gnn_d=model["n_gnn_output_features"])
    print("Zoonoformer")
    print(model)
    print("Number of parameters = ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model
