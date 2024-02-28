import torch.nn as nn
from models.nlp.embedding.embedding import EmbeddingLayer, ConvolutionEmbeddingLayer
from models.nlp.transformer.encoder import EncoderLayer, Encoder
from models.nlp.transformer.decoder import DecoderLayer, Decoder
from utils import nn_utils


# self implementation

# encoder-decoder architecture
class Transformer(nn.Module):
    def __init__(self, n_tokens, max_seq_len, n_classes, N=6, input_dim=512, hidden_dim=1024, h=8):
        super(Transformer, self).__init__()
        self.source_embedding = EmbeddingLayer(vocab_size=n_tokens, max_seq_len=max_seq_len, dim=input_dim)
        self.target_embedding = EmbeddingLayer(vocab_size=n_tokens, max_seq_len=max_seq_len, dim=input_dim)
        self.encoder = Encoder(EncoderLayer(h, input_dim, hidden_dim), N)
        self.decoder = Decoder(DecoderLayer(h, input_dim, hidden_dim), N)

    def forward(self, source, target, source_mask, target_mask):
        source_emb = self.source_embedding(source)
        source_emb = self.encoder(X=source_emb,
                                  mask=source_mask)
        decoder_emb = self.decoder(X=target,
                                   source_emb=source_emb,
                                   source_mask=source_mask,
                                   target_mask=target_mask)

        return decoder_emb


# only encoder
class TransformerEncoder(nn.Module):
    def __init__(self, n_tokens, max_seq_len, N=6, input_dim=512, hidden_dim=1024, h=8):
        super(TransformerEncoder, self).__init__()
        self.embedding = EmbeddingLayer(vocab_size=n_tokens, max_seq_len=max_seq_len, dim=input_dim)
        self.encoder = Encoder(EncoderLayer(h, input_dim, hidden_dim), N)

    def forward(self, X, mask):
        X = self.embedding(X)
        X = self.encoder(X, mask)
        return X


def get_transformer_encoder(model):
    tf_model = TransformerEncoder(n_tokens=model["n_tokens"],
                                  max_seq_len=model["max_seq_len"],
                                  N=model["depth"],
                                  input_dim=model["input_dim"],
                                  hidden_dim=model["hidden_dim"],
                                  h=model["n_heads"])
    print(tf_model)
    print("Number of parameters = ", sum(p.numel() for p in tf_model.parameters() if p.requires_grad))
    return tf_model.to(nn_utils.get_device())


# encoder classifier
class TransformerEncoderClassifier(nn.Module):
    def __init__(self, n_tokens, max_seq_len, N=6, input_dim=512, hidden_dim=1024, h=8, n_classes=None):
        super(TransformerEncoderClassifier, self).__init__()
        self.embedding = EmbeddingLayer(vocab_size=n_tokens, max_seq_len=max_seq_len, dim=input_dim)
        self.encoder = Encoder(EncoderLayer(h, input_dim, hidden_dim), N)
        # last linear layer: input_dim--> n_classes
        self.linear_output = nn.Linear(input_dim, n_classes)

    def forward(self, X, mask):
        X = self.embedding(X)
        X = self.encoder(X, mask)

        # pool the model embeddings of all tokens in the input sequence using mean
        X = X.mean(dim=1)

        # embedding to be used for interpretability
        self.tf_enc_embedding = X

        y = self.linear_output(X)
        return y


def get_transformer_encoder_classifier(model):
    tf_model = TransformerEncoderClassifier(n_tokens=model["n_tokens"],
                                            max_seq_len=model["max_seq_len"],
                                            N=model["depth"],
                                            input_dim=model["input_dim"],
                                            hidden_dim=model["hidden_dim"],
                                            h=model["n_heads"],
                                            n_classes=model["n_classes"])
    print(tf_model)
    print("Number of parameters = ", sum(p.numel() for p in tf_model.parameters() if p.requires_grad))
    return tf_model.to(nn_utils.get_device())
