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

# +
# Load dataset
sequence_settings = {
    "batch_size": 1,
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

# +
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

virprobert_model.self_attn.self_attn.shape

WIV04_idx = 0
inter_seg_attn = virprobert_model.self_attn.self_attn[WIV04_idx, :, :, :]

inter_seg_attn.shape

pos_mapping = {}
j = 0
for i in range(0, len(wiv04_input_df["seq"][0])+1, 64):
    start = i+1
    # end = seq_len if i+127 > seq_len else i+127
    end = i + 256
    pos_mapping[j] = f"{j}: {start}-{end}"
    j += 1

pos_mapping

# +
plt.clf()
plt.rcParams["xtick.labelsize"] = 40
plt.rcParams["ytick.labelsize"] = 40
plt.rcParams.update({'font.size': 40})
fig, axs = plt.subplots(4, 2, figsize=(80, 100), sharex=False, sharey=True)

c = 0
for i in range(4):
    for j in range(2):
        df = pd.DataFrame(inter_seg_attn[c].squeeze().detach().cpu().numpy())
        df.rename(columns=pos_mapping, inplace=True)
        df.rename(index=pos_mapping, inplace=True)
        sns.heatmap(df, ax=axs[i, j], linewidth=.1)
        axs[i, j].set_title(f"Head {c}")
        c += 1

plt.tight_layout(pad=.1)
plt.show()
# -

inter_seg_attn.mean(dim=0).shape

plt.clf()
plt.figure(figsize=(8, 8))
sns.set_theme()
sns.set_style("whitegrid")
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams.update({'font.size': 12})
sns.heatmap(inter_seg_attn.mean(dim=0).detach().cpu().numpy(), linewidth=.1, cmap="crest")
plt.show()

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


