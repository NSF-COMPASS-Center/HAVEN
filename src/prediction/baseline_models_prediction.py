import os
import pandas as pd

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

from utils import kmer_utils, utils, visualization_utils
from prediction.models.baseline import logistic_regression
from prediction.models.baseline import random_forest


def execute(input_settings, output_settings, classification_settings):
    # input settings
    input_dir = input_settings["input_dir"]
    inputs = input_settings["file_names"]

    # output settings
    output_dir = output_settings["output_dir"]
    results_dir = output_settings["results_dir"]
    visualizations_dir = output_settings["visualizations_dir"]
    sub_dir = output_settings["sub_dir"]
    output_prefix = output_settings["prefix"]
    output_prefix = "_" + output_prefix if output_prefix is not None else ""

    # classification settings
    k = classification_settings["kmer_settings"]["k"]
    classification_type = classification_settings["type"]
    models = classification_settings["models"]
    id_col = classification_settings["id_col"]
    sequence_col = classification_settings["sequence_col"]

    label_settings = classification_settings["label_settings"]
    label_col = label_settings["label_col"]

    results = {}
    feature_importance = {}
    validation_scores = {}
    itr = 0
    for input in inputs:
        print(f"Iteration {itr}")
        # 1. Read the data files
        train_df, test_df = read_dataset(input_dir, input, id_col, sequence_col, label_col)

        # X_train_resampled, y_train_resampled = utils.random_oversampling(X_train, y_train)
        # return X_train_resampled, X_test, y_train_resampled, y_test

        # pos_df = train_df[train_df[label_col] == 'Homo sapiens']
        # neg_df = train_df[train_df[label_col] != 'Homo sapiens']
        # print(f"Number of positive samples = {pos_df.shape}")
        # print(f"Number of negative samples = {neg_df.shape}")
        # pos_df = test_df[test_df[label_col] == 'Homo sapiens']
        # neg_df = test_df[test_df[label_col] != 'Homo sapiens']
        # print(f"Number of positive samples = {pos_df.shape}")
        # print(f"Number of negative samples = {neg_df.shape}")


        # downsampling number of positives
        # print("Downsampling training dataset >")
        # print(f"Training dataset size before downsampling = {train_df.shape}")
        # pos_df = train_df[train_df[label_col] == 'Homo sapiens']
        # neg_df = train_df[train_df[label_col] != 'Homo sapiens']
        # print(f"Number of positive samples = {pos_df.shape}")
        # print(f"Number of negative samples = {neg_df.shape}")
        # train_df = pd.concat([pos_df[:neg_df.shape[0]], neg_df])
        # train_df = train_df[~train_df.index.duplicated()]
        #
        # print("Downsampling testing dataset >")
        # print(f"Testing dataset size before downsampling = {test_df.shape}")
        # pos_df = test_df[test_df[label_col] == 'Homo sapiens']
        # neg_df = test_df[test_df[label_col] != 'Homo sapiens']
        # print(f"Number of positive samples = {pos_df.shape}")
        # print(f"Number of negative samples = {neg_df.shape}")
        # test_df = pd.concat([pos_df[:neg_df.shape[0]], neg_df])
        # test_df = test_df[~test_df.index.duplicated()]
        # print("==========")
        # print(f"Training dataset size after downsampling= {train_df.shape}")
        # print(f"Testing dataset size after downsampling= {test_df.shape}")

        df = pd.concat([train_df, test_df])

        # 2. filter out noise: labels configured to be excluded, NaN labels
        df = utils.filter_noise(df, label_settings)

        # 3. Compute kmer features
        kmer_df = kmer_utils.compute_kmer_features(df, k, id_col, sequence_col, label_col)
        print(f"back in prediction_with_inputs_split = {kmer_df.shape}")

        # get the split column again to distinguish train and test datasets
        kmer_df = kmer_df.join(df["split"], on=id_col, how="left")
        print(f"kmer_df size after join with split on id = {kmer_df.shape}")

        # 4. Group the labels (if applicable) and convert the string labels to mapped integer indices
        kmer_df_with_transformed_label, idx_label_map = utils.transform_labels(kmer_df, label_settings, classification_type)
        print(f"kmer_df_with_transformed_label size = {kmer_df_with_transformed_label.shape}")

        # 5. Perform classification
        X_train, X_test, y_train, y_test = get_standardized_datasets(kmer_df_with_transformed_label, label_col=label_col)

        for model in models:
            if model["active"] is False:
                print(f"Skipping {model['name']} ...")
                continue
            model_name = model["name"]
            if model_name not in results:
                # first iteration
                results[model_name] = []
                feature_importance[model_name] = []
                validation_scores[model_name] = []

            # Set necessary values within model object for cleaner code and to avoid passing multiple arguments.
            model["label_col"] = label_col
            model["classification_type"] = classification_type

            if model["name"] == "lr":
                print("Executing Logistic Regression")
                y_pred, feature_importance_df, validation_scores_df = logistic_regression.run(X_train, X_test, y_train, model)
            elif model["name"] == "rf":
                print("Executing Random Forest")
                y_pred, feature_importance_df, validation_scores_df = random_forest.run(X_train, X_test, y_train, model)
            else:
                continue

            #  Create the result dataframe and remap the class indices to original input labels
            result_df = pd.DataFrame(y_pred)
            result_df["y_true"] = y_test.values
            result_df.rename(columns=idx_label_map, inplace=True)
            result_df["y_true"] = result_df["y_true"].map(idx_label_map)
            result_df["itr"] = itr

            # Remap the class indices to original input labels
            feature_importance_df.rename(index=idx_label_map, inplace=True)
            feature_importance_df["itr"] = itr

            validation_scores_df["itr"] = itr

            results[model_name].append(result_df)
            feature_importance[model_name].append(feature_importance_df)
            validation_scores[model_name].append(validation_scores_df)
        itr += 1

    # write the raw results in csv files
    output_filename_prefix = f"kmer_k{k}_{label_col}_{classification_type}_presplit" + output_prefix + "_"
    output_results_dir = os.path.join(output_dir, results_dir, sub_dir)
    utils.write_output(results, output_results_dir, output_filename_prefix, "output",)
    utils.write_output(feature_importance, output_results_dir, output_filename_prefix, "feature_imp")
    utils.write_output(validation_scores, output_results_dir, output_filename_prefix, "validation_scores")

    # create plots for validation scores
    plot_validation_scores(validation_scores, os.path.join(output_dir, visualizations_dir, sub_dir), output_filename_prefix)


def read_dataset(input_dir, input, id_col, sequence_col, label_col):
    train_datasets = []
    test_datasets = []
    sub_dir = input["dir"]

    train_files = input["train"]
    for train_file in train_files:
        input_file_path = os.path.join(input_dir, sub_dir, train_file)
        df = pd.read_csv(input_file_path, usecols=[id_col, sequence_col, label_col])
        print(f"input train file: {input_file_path}, size = {df.shape}")
        train_datasets.append(df)

    test_files = input["test"]
    for test_file in test_files:
        input_file_path = os.path.join(input_dir, sub_dir, test_file)
        df = pd.read_csv(input_file_path, usecols=[id_col, sequence_col, label_col])
        print(f"input test file: {input_file_path}, size = {df.shape}")
        test_datasets.append(df)

    train_df = pd.concat(train_datasets)
    train_df["split"] = "train"
    train_df.set_index(id_col, inplace=True)
    print(f"Size of input train dataset = {train_df.shape}")

    test_df = pd.concat(test_datasets)
    test_df["split"] = "test"
    test_df.set_index(id_col, inplace=True)
    print(f"Size of input test dataset = {test_df.shape}")

    return train_df, test_df


def get_standardized_datasets(df, label_col):
    drop_cols = ["split", label_col]

    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df[label_col]

    X_test = test_df.drop(columns=drop_cols)
    y_test = test_df[label_col]

    # Standardize dataset
    min_max_scaler = MinMaxScaler()
    min_max_scaler_fit = min_max_scaler.fit(X_train)
    feature_names = min_max_scaler_fit.get_feature_names_out()

    # creating a pandas df with column headers = feature names so that the prediction model can also have the same order of feature names
    # this is needed while getting the feature_importances from the model and mapping it back to the feature names.
    X_train = pd.DataFrame(min_max_scaler_fit.transform(X_train), columns=feature_names)
    X_test = pd.DataFrame(min_max_scaler_fit.transform(X_test), columns=feature_names)

    return X_train, X_test, y_train, y_test


def plot_validation_scores(model_dfs, output_dir, output_filename_prefix):
    for model_name, dfs in model_dfs.items():
        df = pd.concat(dfs)

        cols = list(df.columns)
        # all columns without the itr column
        cols.remove("itr")

        # plot only for the first iteration
        df = df[df["itr"] == 0]
        df.drop(columns=["itr"], inplace=True)
        df["split"] = range(1, 6)
        transformed_df = pd.melt(df, id_vars=["split"])
        output_file_name = output_filename_prefix.format(model_name) + "validation_scores.png"
        output_file_path = os.path.join(output_dir, output_file_name)
        # create any missing parent directories
        Path(os.path.dirname(output_file_path)).mkdir(parents=True, exist_ok=True)
        visualization_utils.validation_scores_multiline_plot(transformed_df, output_file_path)
