import torch
from utils import visualization_utils
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

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
