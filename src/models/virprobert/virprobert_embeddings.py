import torch.nn as nn
from models.baseline.nlp.transformer.multi_head_attention import MultiHeadAttention
from models.baseline.nlp.transformer.feed_forward_layer import FeedForwardLayer
from utils import nn_utils, constants
import torch
import torch.nn.functional as F
from models.protein_sequence_classification import ProteinSequenceClassification
import tqdm


class VirProBERT_Emb(ProteinSequenceClassification):
    def __init__(self, pre_trained_model, segment_len, cls_token, h=8, input_dim=512, hidden_dim=2048, stride=1, n_mlp_layers=2, n_classes=1):
        super(VirProBERT_Emb, self).__init__(input_dim, hidden_dim, n_mlp_layers, n_classes, batch_norm=True)
        self.pre_trained_model = pre_trained_model
        self.input_dim = input_dim
        self.self_attn = MultiHeadAttention(h, input_dim)
        self.feed_forward = FeedForwardLayer(input_dim, hidden_dim)
        self.segment_len = segment_len
        self.cls_token = cls_token
        self.stride = stride

    # def forward(self, X, embedding_only = True):
    #     self.input_embedding = get_embedding(X)
    #     print("TEST  - FORWARD IN VIRPROBERT")
    #     return self.input_embedding
    #     # return super().forward(X, embedding_only = embedding_only)

    def get_embedding(self, X):
        # X: b x n where n is the maximum sequence length in the batch
        # 1. split into segments
        batch_size = X.shape[0]  # batch_size
        # split each sequence into smaller segments along the dimension of the sequence length
        # let # segment = n_s (depending on segment_len and stride)
        X = X.unfold(dimension=1, size=self.segment_len, step=self.stride)  # b x n_s x segment_len

        # reshape the tensor to individual sequences of segment_len diregarding the sequence dimension (i.e. batch)
        # since we only need to generate embeddings for each segment where the sequence identity does not matter
        # contiguous ensures contiguous memory allocation for every value in the tensor
        # this will enable reshaping using view which only changes the shape(view) of the tensor without creating a copy
        # reshape() 'might' create a copy. Hence, we use view() to save memory
        X = X.contiguous().view(-1, self.segment_len)  # (b * n_s) x segment_len

        if self.cls_token:
            # add cls token to the beginning to every segment
            cls_tokens = torch.full(size=(X.shape[0], 1), fill_value=constants.CLS_TOKEN_VAL,
                                    device=nn_utils.get_device())
            X = torch.cat([cls_tokens, X], dim=1)

        # generate embeddings
        X = self.pre_trained_model(X, mask=None)  # (b * n_s) x segment_len x input_dim

        # reshape back into sequences of chunks, i.e. re-introduce the batch dimension.
        # here -1 will account for n_s which changes with the sequence length in every batch
        # we use segment_len + 1 to account for the added CLS token

        if self.cls_token:
            X = X.view(batch_size, -1, self.segment_len + 1, self.input_dim)  # b x n_s x segment_len + 1 x input_dim
        else:
            X = X.view(batch_size, -1, self.segment_len, self.input_dim)  # b x n_s x segment_len + 1 x input_dim

        if self.cls_token:
            # OPTION 1: representative vector for each segment = CLS token embedding in every segment
            X = X[:, :, 0, :]
        else:
            # OPTION 2: representative vector for each segment = mean of the embeddings of tokens in every segment
            # mean along segment_len dimension, i.e dim=2
            X = X.mean(dim=2)  # b x n_s x input_dim

        # attention between the chunks in every sequence
        X = self.self_attn(X, X, X)  # b x n_s x input_dim

        # feed-forward for projection + non-linear activation
        X = self.feed_forward(X)  # b x n_s x input_dim

        # pool the embeddings of all segments in the input sequence using mean to generate a vector embedding for each sequence
        X = X.mean(dim=1)  # b x input_dim
        return X

    def get_model(model_params) -> ProteinSequenceClassification:
        model = VirProBERT_Emb(pre_trained_model=model_params["pre_trained_model"],
                           segment_len=model_params["segment_len"],
                           cls_token=model_params["cls_token"],
                           h=model_params["n_heads"],
                           input_dim=model_params["input_dim"],
                           hidden_dim=model_params["hidden_dim"],
                           n_mlp_layers=model_params["n_mlp_layers"],
                           stride=model_params["stride"],
                           n_classes=model_params["n_classes"])
        print(model)
        print("VirProBERT_Emb: Number of parameters = ", sum(p.numel() for p in model.parameters() if p.requires_grad))
        print("EMBEDDINGS 2.0")

        return ProteinSequenceClassification.return_model(model, model_params["data_parallel"])

