import torch.nn as nn
from models.nlp.embedding.embedding import EmbeddingLayer, ConvolutionEmbeddingLayer
from models.nlp.transformer.encoder import EncoderLayer, Encoder
from models.nlp.transformer.decoder import DecoderLayer, Decoder
from models.nlp.transformer.generator import Generator
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
        self.generator = Generator(d=hidden_dim, vocab_size=n_tokens)

    def forward(self, source, target, source_mask, target_mask):
        source_emb = self.source_embedding(source)
        source_emb = self.encoder(X=source_emb,
                                  mask=source_mask)
        decoder_emb = self.decoder(X=target,
                                   source_emb=source_emb,
                                   source_mask=source_mask,
                                   target_mask=target_mask)

        # generate the next token using the decoder_emb
        y = self.generator(decoder_emb)
        return y

# only encoder
class TransformerEncoder(nn.Module):
    def __init__(self, n_tokens, max_seq_len, n_classes, N=6, input_dim=512, hidden_dim=1024, h=8):
        super(TransformerEncoder, self).__init__()
        self.embedding = EmbeddingLayer(vocab_size=n_tokens, max_seq_len=max_seq_len, dim=input_dim)
        self.encoder = Encoder(EncoderLayer(h, input_dim, hidden_dim), N)
        self.linear = nn.Linear(input_dim, n_classes)

    def forward(self, X):
        X = self.embedding(X)
        X = self.encoder(X)
        # pool the embeddings of all tokens using mean
        X = X.mean(dim=1)

        y = self.linear(X)
        return y


class Transformer_Conv1D(Transformer):
    def __init__(self, n_tokens, kernel_size, stride, padding, max_seq_len, n_classes, N=6, input_dim=512,
                 hidden_dim=1024, h=8):
        super(Transformer_Conv1D, self).__init__(n_tokens, max_seq_len, n_classes, N, input_dim, hidden_dim, h)
        self.embedding = ConvolutionEmbeddingLayer(n_tokens, max_seq_len, input_dim, kernel_size, stride, padding)


def get_transformer_model(model):
    if model["with_convolution"]:
        tf_model = Transformer_Conv1D(n_tokens=model["n_tokens"],
                                      kernel_size=model["kernel_size"],
                                      stride=model["stride"],
                                      padding=model["padding"],
                                      max_seq_len=model["max_seq_len"],
                                      n_classes=model["n_classes"],
                                      N=model["depth"],
                                      input_dim=model["input_dim"],
                                      hidden_dim=model["hidden_dim"],
                                      h=model["n_heads"])
    else:
        tf_model = Transformer(n_tokens=model["n_tokens"],
                               max_seq_len=model["max_seq_len"],
                               n_classes=model["n_classes"],
                               N=model["depth"],
                               input_dim=model["input_dim"],
                               hidden_dim=model["hidden_dim"],
                               h=model["n_heads"])

    print(tf_model)
    print("Number of parameters = ", sum(p.numel() for p in tf_model.parameters() if p.requires_grad))
    return tf_model.to(nn_utils.get_device())
