from torch import nn
import torch.nn.functional as F
from utils import nn_utils, constants
import torch
from models.protein_sequence_classification import ProteinSequenceClassification


class VirProBERT_wo_HierAttn(ProteinSequenceClassification):
    def __init__(self, pre_trained_model, segment_len, cls_token, input_dim, hidden_dim, stride=1, n_mlp_layers=2, n_classes=1):
        super(VirProBERT_wo_HierAttn, self).__init__(input_dim, hidden_dim,
                                                     n_mlp_layers=n_mlp_layers,
                                                     n_classes=n_classes,
                                                     batch_norm=False)
        self.pre_trained_model = pre_trained_model
        self.segment_len = segment_len
        self.cls_token = cls_token
        self.stride = stride
        self.input_dim = input_dim

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

        # pool the embeddings of all segments in the input sequence using mean to generate a vector embedding for each sequence
        X = X.mean(dim=1)  # b x input_dim

        return X

    # def forward() : use the template implementation in ProteinSequenceClassification

    def get_model(model_params) -> ProteinSequenceClassification:
        model = VirProBERT_wo_HierAttn(pre_trained_model=model_params["pre_trained_model"],
                                       input_dim=model_params["input_dim"],
                                       cls_token=model_params["cls_token"],
                                       stride=model_params["stride"],
                                       segment_len=model_params["segment_len"],
                                       hidden_dim=model_params["hidden_dim"],
                                       depth=model_params["n_mlp_layers"],
                                       n_classes=model_params["n_classes"])
        print(model)
        print("VirProBERT_wo_HierAttn: Number of parameters = ", sum(p.numel() for p in host_prediction_model.parameters() if p.requires_grad))

        return ProteinSequenceClassification.return_model(model, model_params["data_parallel"])
