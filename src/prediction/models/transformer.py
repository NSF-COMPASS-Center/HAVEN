import torch
from torch import nn
import torch.nn.functional as F
import tqdm
from torch.utils.tensorboard import SummaryWriter


def run(X_train, X_test, y_train, transformer_settings):
    model = Transformer(d=transformer_settings["d"],
                        n_heads=transformer_settings["n"],
                        depth=transformer_settings["depth"],
                        seq_len=transformer_settings["seq_len"],
                        n_tokens=transformer_settings["n_tokens"],
                        n_classes=transformer_settings["n_classes"])
    seq_len_max = transformer_settings["seq_len_max"]
    if torch.cuda.is_available():
        model.cuda()

    summary_writer = SummaryWriter(log_dir=transformer_settings["logging_dir"])  # Tensorboard logging

    optimizer = torch.optim.Adam(transformer_settings["lr"], params=model.parameters())

    # Training
    for e in range(int(transformer_settings["epochs"])):
        print(f"\n Epoch {e}")
        model.train(True)

        for batch in tqdm.tqdm(X_train):
            optimizer.zero_grad()

            X = batch.text[0]
            y = batch.label - 1

            if X.size(1) > seq_len_max:
                X = X[:, :seq_len_max]
            output = model(X)
            loss = F.nll_loss(output, y)

            loss.backward()
            optimizer.step()

            summary_writer.add_scalar("Training loss", float(loss.item()))

    with torch.no_grad():
        model.train(False)

        for batch in tqdm.tqdm(X_test):
            X = batch.text[0]
            y = batch.label - 1

            if X.size(1) > seq_len_max:
                X = X[:, :seq_len_max]
            output = model(X)

            summary_writer.add_scalar("Test loss", float(loss.item()))


class SelfAttention(nn.Module):
    def __init__(self, d, n_heads=8):
        """
        :param d: output dimension
        :param n_heads: number of attention heads
        """
        super().__init__()
        self.d = d
        self.n_heads = n_heads

        self.attn_queries == nn.Linear(d, d * n_heads, bias=False)
        self.attn_keys == nn.Linear(d, d * n_heads, bias=False)
        self.attn_values == nn.Linear(d, d * n_heads, bias=False)


        # combine outputs of multiple heads into one single d-dimensional vector
        self.combine_heads == nn.Linear(d * n_heads, d)


    def forward(self, X):
        # b: minibatch size
        # n: number of sequences
        b, n, _ = X.size()

        queries = self.attn_queries(X).view(b, n, self.n_heads, self.d)
        keys = self.attn_keys(X).view(b, n, self.n_heads, self.d)
        values = self.attn_values(X).view(b, n, self.n_heads, self.d)

        # reshape heads into batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * self.n_heads, n, self.d)
        queries = queries.transpose(1, 2).contiguous().view(b * self.n_heads, n, self.d)
        values = values.transpose(1, 2).contiguous().view(b * self.n_heads, n, self.d)

        # scaling
        queries = queries / (self.d ** (1/4))
        keys = keys / (self.d ** (1 / 4))

        # dot product of queries and keys to get attention weights
        # attn_weights shape = b * d, n, n containing the raw weights
        attn_weights = torch.bmm(queries, keys.transpose(1,2))

        # normalize weights
        attn_weights = F.softmax(attn_weights, dim=2)

        # apply self-attention to the values
        output = torch.bmm(attn_weights, values).view(b, self.n_heads, n, self.d)

        # combine the outputs from the heads
        output = output.transpose(1, 2).contiguous().view(b, n, self.n_heads * self.d)
        return self.combine_heads(output)



class TransformerLayer(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.attention = SelfAttention(d, n_heads)

        self.norm_1 = nn.LayerNorm(d)
        self.norm_2 = nn.LayerNorm(d)

        # h: hidden layer dimension multiplicative factor
        self.h = 4

        self.feed_forward = nn.Sequential(
            nn.Linear(d, self.h * d),
            nn.ReLU(),
            nn.Linear(self.h * d, d)
        )


    def forward(self, X):
        attn_out = self.attention(X)
        norm_out_1 = self.norm_1(attn_out + X)
        feed_forward_out = self.feed_forward(norm_out_1)
        return self.norm_2(feed_forward_out + norm_out_1)


class Transformer(nn.Module):
    def __init__(self, d, n_heads, depth, seq_len, n_tokens, n_classes):
        super().__init__()

        self.n_tokens = n_tokens
        self.token_emb = nn.Embedding(n_tokens, d)
        self.pos_emb = nn.Embedding(seq_len, d)

        transformer_layers = []
        for i in range(depth):
            transformer_layers.append(TransformerLayer(d=d, n_heads=n_heads))

        self.transformer_layers = nn.Sequential(*transformer_layers)

        # map output embedding to class logits
        self.output_probabilities = nn.Linear(d, n_classes)


    def forward(self, X):
        """
        :param X: (b, n) tensor of integer values representing amino acid tokens
        :return: (b, n_classes) tensor of log probabilities over the classes
        """

        # generate token embeddings
        tokens = self.token_emb(X)
        b, n, d = tokens.size()

        # generate position embeddings
        positions = torch.arrange(n)
        positions = self.pos_emb(positions)[None, :, :].expand(b, n, d)

        X_emb = tokens + positions
        X_out = self.transformer_layers(X_emb)

        # average pool over n dimension and project to class probabilities
        X_out_probs = self.output_probabilities(X_out.mean(dim=1))
        return F.log_softmax(X_out_probs, dim=1)
