# +
import sys
import os
from pathlib import Path
sys.path.append(os.path.join(os.getcwd(), "..", "..", ".."))
sys.path.append(os.path.join(os.getcwd(), "..", "..", "..", ".."))
sys.path.append(os.path.join(os.getcwd(), "..", "..", "..", "..", ".."))
sys.path.append(os.path.join(os.getcwd(), "..", ".."))

#sys.path.append(os.path.join(os.getcwd(), ".."))
sys.path
# -

import torch
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.models.baseline.nlp.transformer.transformer import TransformerEncoder
from src.models.virprobert import VirProBERT
from src.utils import constants, nn_utils, utils
from src.datasets.protein_sequence_with_id_dataset import ProteinSequenceDatasetWithID
from src.datasets.collations.padding_with_id import PaddingWithID


input_file_path = os.path.join(os.getcwd(), "..","..", "..", "input/data/coronaviridae/20240313/sarscov2/uniprot/variants/sarscov2_variants_s.csv")
input_df = pd.read_csv(input_file_path)
input_df

wiv04_input_df = input_df[input_df["id"] == "WIV04"]
wiv04_input_df

# ### Load the model

# +
# Transformer Encoder

pre_train_encoder_settings = {
    "n_heads": 8,
    "depth": 6,
    "input_dim": 512, # input embedding dimension
    "hidden_dim": 1024,
    "max_seq_len": 256,
    "cls_token": True,
    "vocab_size": constants.VOCAB_SIZE
}

pre_trained_encoder_model = TransformerEncoder.get_transformer_encoder(pre_train_encoder_settings, pre_train_encoder_settings["cls_token"])

# +
# VirProBERT model
virprobert_settings = {
    "n_mlp_layers": 2,
    "n_classes": 8,
    "input_dim": 512, # input embedding dimension,
    "hidden_dim": 1024,
    "n_heads": 8,
    "stride": 64,
    "cls_token": True,
    "segment_len": pre_train_encoder_settings["max_seq_len"],
    "data_parallel": False,
    "pre_trained_model": pre_trained_encoder_model
}

virprobert_model = VirProBERT.get_model(virprobert_settings)

model_path = os.path.join(os.getcwd(), "..","..", "..", "output/raw/coronaviridae_s_prot_uniref90_embl_vertebrates_t0.01_c8/20240828/host_multi/fine_tuning_hybrid_cls/mlm_tfenc_l6_h8_lr1e-4_uniref90viridae_msl256b512_ae_bn_vs30cls_s64_hybrid_attention_s64_fnn_2l_d1024_lr1e-4_itr4.pth")
virprobert_model.load_state_dict(torch.load(model_path, map_location=nn_utils.get_device()))
# -

# ProstT5 model
prostT5_settings = {
    "pre_trained_model_link": "Rostlab/ProstT5"
      "hugging_face_cache_dir": "output/cache_dir"
      loss: "FocalLoss"
      n_mlp_layers: 2
      n_classes: 8
      input_dim: 1024 # input embedding dimension
      hidden_dim: 1024
      data_parallel: False
}

# +
# Load dataset
sequence_settings = {
    "batch_size": 16,
    "id_col": "id",
    "sequence_col": "seq",
    "truncate": False,
    "split": False,
    "feature_type": "token"
}

label_settings = {
    "label_col": "virus_host_name",
    "exclude_labels": [ "nan"],
    "label_groupings": {
      "Chicken": [ "gallus gallus" ],
      "Human": [ "homo sapiens" ],
      "Cat": [ "felis catus" ],
      "Pig": [ "sus scrofa" ],
      "Gray wolf": [ "canis lupus" ],
      "Horshoe bat": ["rhinolophus sp."],
      "Ferret": ["mustela putorius"],
      "Chinese rufous horseshoe bat": ["rhinolophus sinicus"],
    }
}
input_df, index_label_map = utils.transform_labels(input_df, label_settings, classification_type="multi")

dataset = ProteinSequenceDatasetWithID(input_df, 
                                       id_col=sequence_settings["id_col"], 
                                       sequence_col=sequence_settings["sequence_col"], 
                                       max_seq_len=pre_train_encoder_settings["max_seq_len"], 
                                       truncate=sequence_settings["truncate"], 
                                       label_col=label_settings["label_col"])
    
dataset_loader = DataLoader(dataset=dataset, 
                            batch_size=sequence_settings["batch_size"], 
                            shuffle=True,
                            collate_fn=PaddingWithID(pre_train_encoder_settings["max_seq_len"]))
# -

virprobert_model.eval()
seq_id, seq, label = next(iter(dataset_loader))

print(f"seq_id = {seq_id}")
print(f"seq = {seq}, seq_len = {seq.shape}")
print(f"label = {label}")

output = virprobert_model(seq)

output

output_prob = F.softmax(output, dim=-1)
result_df = pd.DataFrame(output_prob.detach().cpu().numpy())
result_df["id"] = seq_id
result_df["y_true"] = label.detach().cpu().numpy()
result_df

# +
wiv04_input_df, index_label_map = utils.transform_labels(wiv04_input_df, label_settings, classification_type="multi")

dataset = ProteinSequenceDatasetWithID(wiv04_input_df, 
                                       id_col=sequence_settings["id_col"], 
                                       sequence_col=sequence_settings["sequence_col"], 
                                       max_seq_len=pre_train_encoder_settings["max_seq_len"], 
                                       truncate=sequence_settings["truncate"], 
                                       label_col=label_settings["label_col"])
    
dataset_loader = DataLoader(dataset=dataset, 
                            batch_size=sequence_settings["batch_size"], 
                            shuffle=True,
                            collate_fn=PaddingWithID(pre_train_encoder_settings["max_seq_len"]))
# -

virprobert_model.eval()
seq_id, seq, label = next(iter(dataset_loader))

print(f"seq_id = {seq_id}")
print(f"seq = {seq}, seq_len = {seq.shape}")
print(f"label = {label}")

output = virprobert_model(seq)
output

output_prob = F.softmax(output, dim=-1)
result_df = pd.DataFrame(output_prob.detach().cpu().numpy())
result_df["id"] = seq_id
result_df["y_true"] = label.detach().cpu().numpy()
result_df


def get_mean_attention_values(tf_model):
    # attention values of the last encoder layer
    attn_values = tf_model.encoder.layers[5].self_attn.self_attn.squeeze()
    return torch.mean(attn_values, dim=0)

# analyze the attention values of all sequences in a dataset
def analyze_dataset_attention_values(tf_model, dataset_loader, seq_max_length):
    attn_dfs = []
    for _, record in enumerate(dataset_loader):
        seq, label = record

        # compute actual (unpadded) length of sequence
        seq_len = torch.count_nonzero(seq).item()
        if seq_len < seq_max_length:
            continue

        tf_model(seq)
        mean_attn_values = get_mean_attention_values(tf_model)
        mean_of_mean_attn_values = torch.mean(mean_attn_values, dim=0, keepdim=True)
        attn_dfs.append(mean_of_mean_attn_values.cpu().detach().numpy())

    attn_df = np.concatenate(attn_dfs, axis=0)
    visualization_utils.heat_map(attn_df)
    return attn_df

def analyze_sequence_attention_values(tf_model, sample_seq, sample_label, seq_max_length, idx_amino_acid_map):
    seq_len = torch.count_nonzero(sample_seq)
    print(sample_seq.shape)
    print(f"Sequence length = {seq_len}")

    sample_pred = torch.argmax(F.softmax(tf_model(sample_seq), dim=1), dim=1)
    print(f"Label = {index_label_map[sample_label_label.item()]}")
    print(f"Prediction = {index_label_map[sample_pred.item()]}")
    mean_attn_values = get_mean_attention_values(tf_model)

    plot_mean_attention_values(mean_attn_values, seq=sample_seq, seq_len=seq_len)
    plot_mean_of_mean_attention_values(mean_attn_values, seq=sample_seq,
                                       seq_len=seq_len, seq_max_length=seq_max_length,
                                       idx_amino_acid_map=idx_amino_acid_map)


def plot_mean_attention_values(x, seq=None, seq_len=None, idx_amino_acid_map=None):
    ticklabels = seq.cpu().detach().numpy().squeeze()[:seq_len]
    ticklabels_mapped = [idx_amino_acid_map[x] for x in ticklabels]

    plt.rcParams['xtick.labelsize'] = 5
    plt.rcParams['ytick.labelsize'] = 5
    plt.figure(figsize=(12, 12))
    data = x.cpu().detach().numpy()

    sns.heatmap(data=data[:seq_len, :seq_len], xticklabels=ticklabels_mapped, yticklabels=ticklabels_mapped)
    plt.show()


def plot_mean_of_mean_attention_values(x, seq=None, seq_len=None, seq_max_length=None):
    tokens = seq.cpu().detach().numpy().squeeze()

    x = torch.mean(x, dim=0)
    df = pd.DataFrame({"tokens": tokens, "attn_vals": x.cpu().detach().numpy(), "pos": range(seq_max_length)})
    df["tokens"] = df["tokens"].map(idx_amino_acid_map)
    df = df.dropna()

    # Top 10 positions with highest attention values
    sorted_df = df.sort_values(by="attn_vals", ascending=False).head(10)
    print("Top 10 tokens + positions with highest attention values for the whole sequence")
    print(sorted_df.head(10))

    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x="pos", y="attn_vals", hue="tokens")
    plt.show()
