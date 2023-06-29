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
    input_file_names = input_settings["file_names"]
    input_split_seeds = input_settings["split_seeds"]

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
    n_iters = classification_settings["n_iterations"]

    label_settings = classification_settings["label_settings"]
    label_col = label_settings["label_col"]

    output_filename_prefix = f"kmer_k{k}_{label_col}_{classification_type}" + output_prefix + "_"
    output_results_dir = os.path.join(output_dir, results_dir, sub_dir)

    results = {}
    feature_importance = {}
    validation_scores = {}
    for iter in range(n_iters):
        print(f"Iteration {iter}")
        # 1. Read the data files
        df = utils.read_dataset(input_dir, input_file_names,
                                cols=[id_col, sequence_col, label_col])
        # 2. Transform labels
        df, index_label_map = utils.transform_labels(df, label_settings,
                                                     classification_type=classification_settings["type"])
        # 3. Split dataset
        train_df, test_df = utils.split_dataset(df, input_split_seeds[iter],
                                                classification_settings["train_proportion"], stratify_col=label_col)

        train_df["split"] = "train"
        test_df["split"] = "test"
        df = pd.concat([train_df, test_df])

        # # 4. filter out noise: labels configured to be excluded, NaN labels
        # df = utils.filter_noise(df, label_settings)

        # 5. Compute kmer features
        kmer_df = kmer_utils.compute_kmer_features(df, k, id_col, sequence_col, label_col)
        print(f"back in prediction_with_inputs_split = {kmer_df.shape}")

        # 6. get the split column again to distinguish train and test datasets
        kmer_df = kmer_df.join(df["split"], on=id_col, how="left")
        print(f"kmer_df size after join with split on id = {kmer_df.shape}")


        # 7. Perform classification
        X_train, X_test, y_train, y_test = get_standardized_datasets(kmer_df, label_col=label_col)

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
                y_pred, feature_importance_df, validation_scores_df, classifier = logistic_regression.run(X_train, X_test, y_train, model)
            elif model["name"] == "rf":
                print("Executing Random Forest")
                y_pred, feature_importance_df, validation_scores_df, classifier = random_forest.run(X_train, X_test, y_train, model)
            else:
                continue

            #  Create the result dataframe and remap the class indices to original input labels
            result_df = pd.DataFrame(y_pred)
            result_df["y_true"] = y_test.values
            result_df.rename(columns=index_label_map, inplace=True)
            result_df["y_true"] = result_df["y_true"].map(index_label_map)
            result_df["itr"] = iter

            # Remap the class indices to original input labels
            feature_importance_df.rename(index=index_label_map, inplace=True)
            feature_importance_df["itr"] = iter

            validation_scores_df["itr"] = iter

            results[model_name].append(result_df)
            feature_importance[model_name].append(feature_importance_df)
            validation_scores[model_name].append(validation_scores_df)

            # write the classification model
            utils.write_output_model(classifier, output_results_dir, output_filename_prefix + f"itr{iter}", model_name)

    # write the raw results in csv files
    utils.write_output(results, output_results_dir, output_filename_prefix, "output",)
    utils.write_output(feature_importance, output_results_dir, output_filename_prefix, "feature_imp")
    utils.write_output(validation_scores, output_results_dir, output_filename_prefix, "validation_scores")

    # create plots for validation scores
    plot_validation_scores(validation_scores, os.path.join(output_dir, visualizations_dir, sub_dir), output_filename_prefix)


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
