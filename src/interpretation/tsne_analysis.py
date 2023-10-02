import torch
import pandas as pd
from sklearn.manifold import TSNE

def get_model_embeddings(model, dataset_loader, model_type):
    model.eval()
    seq_dfs = []
    for _, record in enumerate(dataset_loader):
        seq, label = record
        output = model(seq)
        seq_encoding = get_model_specific_embedding(model, model_type)
        seq_df = pd.DataFrame(seq_encoding.squeeze().cpu().detach().numpy())
        seq_df["label"] = label.squeeze().cpu().detach().numpy()
        seq_dfs.append(seq_df)
    df = pd.concat(seq_dfs)
    return df


def get_model_specific_embedding(model, model_type):
    if model_type == "fnn":
        return model.fnn_emb
    elif model_type == "cnn":
        return model.cnn_emb
    elif model_type == "rnn":
        return model.rnn_emb
    elif model_type == "lstm":
        return model.lstm_emb
    elif model_type == "transformer":
        seq_encoding = model.encoder.encoding
        # embedding = value for each dimension = mean of the dimensional values of all tokens in the input sequence
        return torch.mean(seq_encoding, dim=1, keepdim=True)
    else:
        print(f"ERROR: Unsupported model type - '{model_type}'.\n Supported values 'fnn', 'cnn', 'rnn', 'lstm', and 'transformer'.")


# get tsne embeddings given the emeddings of sequences
def get_tsne_embeddings(df, label_col, n=None):
    # using only the first n columns/ features from the df for computational efficiency
    if n:
        df = df[range(n)]
        print(f"Reduced input shape from {df.shape} to {X.shape}")
    tsne_model = TSNE(n_components=2, verbose=1, init="pca", learning_rate="auto").fit(df)
    X_tsne_emb = pd.DataFrame(tsne_model.fit_transform(df))
    print(f"TSNE embedding shape: {X_tsne_emb.shape}")
    X_tsne_emb[label_col] = df[label_col].values
    return tsne_model, X_tsne_emb