
def compute_baseline_model_embedding(baseline_model, df, label_col, id_col, split_col):
    df = df[df[split_col] == "train"]
    features = baseline_model.feature_names_in_
    coeffs = baseline_model.coef_
    print(f"coeffs shape = {coeffs.shape}")

    features_df = df[features]
    coeff_df = coeffs[df[label_col], :]

    print(f"df.size = {df.shape}")
    print(f"features_df.size = {features_df.shape}")
    print(f"coeff_df.size = {coeff_df.shape}")

    embed_df = features_df.multiply(coeff_df)
    print(f"embed_df shape = {embed_df.shape}")
    embed_df = embed_df.join(df[label_col], on=id_col, how="left")
    print(f"Final embed_df shape = {embed_df.shape}")
    return features_df, coeff_df, embed_df