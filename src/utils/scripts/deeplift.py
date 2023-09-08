#!/usr/src/env python
from captum.attr import Saliency, DeepLift, IntegratedGradients, LayerIntegratedGradients, TokenReferenceBase
import sys
import os
from src.prediction.models.nlp import cnn1d, transformer
from src.utils import utils, nn_utils
import torch
from torch.utils.data import DataLoader
import pandas as pd
from src.prediction.datasets.protein_sequence_dataset import ProteinSequenceDataset
from src.utils.nlp_utils.padding import Padding
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt


def custom_forward(X):
    # with softmax
    return F.softmax(X)

def get_token_dataset_loader(df, sequence_settings, label_col):
    seq_col = sequence_settings["sequence_col"]
    batch_size = sequence_settings["batch_size"]
    max_seq_len = sequence_settings["max_sequence_length"]
    pad_sequence_val = sequence_settings["pad_sequence_val"]
    truncate = sequence_settings["truncate"]
    dataset = ProteinSequenceDataset(df, seq_col, max_seq_len, truncate, label_col)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True,
                      collate_fn=Padding(max_seq_len, pad_sequence_val))


def load_dataset_with_df(df, sequence_settings):
    df = df[[sequence_settings["sequence_col"], label_settings["label_col"]]]
    df, index_label_map = utils.transform_labels(df, label_settings, classification_type="multi")
    dataset_loader = get_token_dataset_loader(df, sequence_settings, label_settings["label_col"])
    return index_label_map, dataset_loader


model_dir = os.path.join(os.getcwd(), "..", "output/raw/coronaviridae/20230813/host_multi_baseline_focal")

model = {"max_seq_len": 1453, "loss": "CrossEntropyLoss", "with_convolution": False, "n_heads": 8, "depth": 4,
         "n_tokens": 27, "n_classes": 7, "n_epochs": 10, "input_dim": 512, "hidden_dim": 1024, "kernel_size": 3,
         "stride": 1}

label_groupings = {
                    "Pig": [ "Sus scrofa (Pig) [TaxID: 9823]" ],
                    "Human": [ "Homo sapiens (Human) [TaxID: 9606]" ],
                    "Wild turkey": [ "Meleagris gallopavo (Wild turkey) [TaxID: 9103]" ],
                    "Japanese pipistrelle": [ "Pipistrellus abramus (Japanese pipistrelle) (Pipistrellus javanicus abramus) [TaxID: 105295]" ],
                    "Lesser bamboo bat": [ "Tylonycteris pachypus (Lesser bamboo bat) (Vespertilio pachypus) [TaxID: 258959]" ],
                    "Chicken": [ "Gallus gallus (Chicken) [TaxID: 9031]" ],
                    "Bovine": [ "Bos taurus (Bovine) [TaxID: 9913]" ]
                }
label_settings = {
    "label_col": "virus_host",
    "exclude_labels": [ "nan"],
    "label_groupings":  label_groupings
}
train_sequence_settings =  {
    "sequence_col": "seq",
    "batch_size": 8,
    "max_sequence_length": 1453,
    "pad_sequence_val": 0,
    "truncate": True,
    "feature_type": "token"
}

test_sequence_settings = train_sequence_settings.copy()
test_sequence_settings["batch_size"] = 1

uniprotkb_coronaviruses_df = pd.read_csv(os.path.join(os.getcwd(), "..", "input/data/coronaviridae/coronaviridae_top_7_hosts.csv"))
uniprotkb_coronaviruses_humans_df = uniprotkb_coronaviruses_df[uniprotkb_coronaviruses_df["virus_host"] == "Homo sapiens (Human) [TaxID: 9606]"]
print(uniprotkb_coronaviruses_humans_df.shape)
index_label_map, coronavirus_dataset_loader = load_dataset_with_df(uniprotkb_coronaviruses_humans_df, test_sequence_settings)


# cnn_model = cnn1d.get_cnn_model(model)
# cnn_model.load_state_dict(torch.load(os.path.join(model_dir, "cnn-l_4_itr0.pth"), map_location=nn_utils.get_device()))
# cnn_model = cnn_model.to(nn_utils.get_device())

model["depth"] = 6
tf_model = transformer.get_transformer_model(model)
tf_model.load_state_dict(torch.load(os.path.join(model_dir, "transformer-l_6-h_8_itr0.pth"), map_location=nn_utils.get_device()))
tf_model = tf_model.to(nn_utils.get_device())

token_reference = TokenReferenceBase(reference_token_idx=1)
reference_seq = token_reference.generate_reference(sequence_length=model["max_seq_len"], device=nn_utils.get_device()).unsqueeze(0)

# dl = LayerIntegratedGradients(cnn_model, cnn_model.embedding)
dl = LayerIntegratedGradients(tf_model, tf_model.embedding)

attributions = []
for i in range(1):
    coronavirus_seq, coronavirus_label = next(iter(coronavirus_dataset_loader))
    attribution = dl.attribute(coronavirus_seq, reference_seq, target=2)
    attribution = attribution.sum(dim=2).squeeze(0)
    attribution = attribution / torch.norm(attribution)
    attributions.append(attribution.numpy())


sns.heatmap(pd.DataFrame(attributions))
plt.show()
