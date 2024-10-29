import torch.nn as nn
from models.baseline.nlp.embedding.embedding import EmbeddingLayer, ConvolutionEmbeddingLayer
from models.baseline.nlp.transformer.encoder import EncoderLayer, Encoder
from models.baseline.nlp.transformer.decoder import DecoderLayer, Decoder
from utils import nn_utils


# self implementation

# encoder-decoder architecture
class Transformer(nn.Module):
    def __init__(self, vocab_size, max_seq_len, n_classes, N=6, input_dim=512, hidden_dim=1024, h=8):
        super(Transformer, self).__init__()
        self.source_embedding = EmbeddingLayer(vocab_size=vocab_size, max_seq_len=max_seq_len, dim=input_dim)
        self.target_embedding = EmbeddingLayer(vocab_size=vocab_size, max_seq_len=max_seq_len, dim=input_dim)
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
    def __init__(self, vocab_size, max_seq_len, N=6, input_dim=512, hidden_dim=1024, h=8):
        super(TransformerEncoder, self).__init__()
        self.embedding = EmbeddingLayer(vocab_size=vocab_size, max_seq_len=max_seq_len, dim=input_dim)
        self.encoder = Encoder(EncoderLayer(h, input_dim, hidden_dim), N)

    def forward(self, X, mask):
        X = self.embedding(X)
        X = self.encoder(X, mask)
        return X

    @staticmethod
    def get_transformer_encoder(model, cls_token=False):
        tf_model = TransformerEncoder(vocab_size=model["vocab_size"],
                                      # adding 1 for CLS token if needed
                                      max_seq_len=model["max_seq_len"] + 1 if cls_token else model["max_seq_len"],
                                      N=model["n_mlp_layers"],
                                      input_dim=model["input_dim"],
                                      hidden_dim=model["hidden_dim"],
                                      h=model["n_heads"])
        print(tf_model)
        print("Number of parameters = ", sum(p.numel() for p in tf_model.parameters() if p.requires_grad))
        return tf_model.to(nn_utils.get_device())