import os
import pandas as pd

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

from utils import utils, dataset_utils, kmer_utils, visualization_utils
from models.baseline.std_ml import svm, random_forest, logistic_regression, xgboost
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# from src.utils.utils import transform_labels
from models.baseline.std_ml.single_linear_layer import SingleLinearLayer



def execute(config):
    # input settings
    input_settings = config["input_settings"]
    input_dir = input_settings["input_dir"]
    file_name = input_settings["file_name"]
    input_file_names = input_settings["emb_file_names"]
    test_file_names = input_settings["emb_test_file_names"]
    input_split_seeds = input_settings["split_seeds"]

    # output settings
    output_settings = config["output_settings"]
    output_dir = output_settings["output_dir"]
    results_dir = output_settings["results_dir"]
    visualizations_dir = output_settings["visualizations_dir"]
    sub_dir = output_settings["sub_dir"]
    output_prefix = output_settings["prefix"]
    output_prefix = output_prefix if output_prefix is not None else ""

    # classification settings
    classification_settings = config["classification_settings"]
    models = classification_settings["models"]
    label_settings = classification_settings["label_settings"]
    sequence_settings = classification_settings["sequence_settings"]
    # kmer_settings = sequence_settings["kmer_settings"]
    n_iters = classification_settings["n_iterations"]
    classification_type = classification_settings["type"]

    id_col = sequence_settings["id_col"]
    sequence_col = sequence_settings["sequence_col"]
    feature_type = sequence_settings["feature_type"]
    label_col = label_settings["label_col"]
    split_col = "split"
    # k = kmer_settings["k"]



    # wandb_config = {
    #     "n_epochs": training_settings["n_epochs"],
    #     "lr": training_settings["max_lr"],
    #     "max_sequence_length": sequence_settings["max_sequence_length"],
    #     "dataset": input_file_names[0]
    # }
    output_filename_prefix = f"{feature_type}_{label_col}_{classification_type}_{output_prefix}"
    output_results_dir = os.path.join(output_dir, results_dir, sub_dir)

    results = {}
    feature_importance = {}
    validation_scores = {}
    convergence = {}
    test_scores = {}
    for iter in range(n_iters):
        print(f"Iteration {iter}")
        # 1. Read the data files
        df = dataset_utils.read_dataset("", file_name, cols=[id_col, sequence_col, label_col])
        input_file_path = os.path.join(input_dir, input_file_names[iter])
        emb_df = pd.read_csv(input_file_path)
        print("Train Embedding data frame: ",input_file_path)
        test_file_path = os.path.join(input_dir, test_file_names[iter])
        print("Test Embedding data frame: ",test_file_path)
        emb_test_df = pd.read_csv(test_file_path)
        # 2. Transform labels
        df, index_label_map = utils.transform_labels(df, label_settings,
                                                     classification_type=classification_type)
        # 3. Split dataset
        train_df, test_df = dataset_utils.split_dataset_stratified(df, input_split_seeds[iter],
                                                                   classification_settings["train_proportion"], stratify_col=label_col)
        y_train = train_df[label_col]
        y_test = test_df[label_col]

        # Standardize the data
        scaler = StandardScaler()
        emb_df_scaled = scaler.fit_transform(emb_df)
        emb_df_scaled = pd.DataFrame(emb_df_scaled, columns=emb_df.columns)
        emb_test_df_scaled = scaler.transform(emb_test_df)
        emb_test_df_scaled = pd.DataFrame(emb_test_df_scaled, columns=emb_test_df.columns)

        # # PCA
        # pca = PCA()
        # pca.fit(emb_df_scaled)
        # explained_variance = pca.explained_variance_ratio_
        # cumulative_variance = explained_variance.cumsum()
        # threshold = 0.95
        # n_components_95 = (cumulative_variance >= threshold).argmax() + 1
        # print("Number of Components to Explain 95% of Variance: ", n_components_95)
        # pca = PCA(n_components=n_components_95)
        # X_train = pca.fit_transform(emb_df_scaled)
        # X_train = pd.DataFrame(X_train)
        # X_test = pca.transform(emb_test_df_scaled)
        # X_test = pd.DataFrame(X_test)


        input_dim = emb_df_scaled.shape[1]
        output_dim = 64

        transform_model = OneLinearLayer(input_dim, output_dim)
        X_train_tensor = torch.tensor(emb_df_scaled, dtype=torch.float32)
        X_test_tensor = torch.tensor(emb_test_df_scaled, dtype=torch.float32)
        X_train = transform_model(X_train_tensor).detach().numpy()
        X_test = transform_model(X_test_tensor).detach().numpy()

        # 5. Perform classification
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
                convergence[model_name] = []
                test_scores[model_name] = []

            # Set necessary values within model_params object for cleaner code and to avoid passing multiple arguments.
            model["label_col"] = label_col
            model["classification_type"] = classification_type

            if "lr" in model_name:
                print("Executing Logistic Regression")
                y_pred, feature_importance_df, validation_scores_df, classifier = logistic_regression.run(X_train, X_test, y_train, model)
            elif "rf" in model_name:
                print("Executing Random Forest")
                y_pred, feature_importance_df, validation_scores_df, classifier = random_forest.run(X_train, X_test, y_train, model)
            elif "svm" in model_name:
                print("Executing Support Vector Machine")
                y_pred, feature_importance_df, validation_scores_df, classifier = svm.run(X_train, X_test, y_train, model)
            elif "xgb" in model_name:
                print("Executing XGBoost")
                y_pred, feature_importance_df, validation_scores_df, classifier = xgboost.run(X_train, X_test, y_train, model)
            else:
                continue

            #  Create the result dataframe and remap the class indices to original input labels
            result_df = pd.DataFrame(y_pred)
            test_scores_df = pd.DataFrame(columns=["test_score", "itr"])
            convergence_df = pd.DataFrame(columns=["convergence", "itr"])
            result_df["y_true"] = y_test.values
            result_df.rename(columns=index_label_map, inplace=True)
            result_df["y_true"] = result_df["y_true"].map(index_label_map)
            result_df["itr"] = iter

            validation_scores_df["itr"] = iter

            results[model_name].append(result_df)
            validation_scores[model_name].append(validation_scores_df)

            # If model_params returns feature importance:
            # Remap the class indices to original input labels
            if feature_importance_df is not None:
                feature_importance_df.rename(index=index_label_map, inplace=True)
                feature_importance_df["itr"] = iter
                feature_importance[model_name].append(feature_importance_df)

            # write the classification model_params
            utils.write_output_model(classifier, output_results_dir, f"{output_filename_prefix}_itr{iter}", model_name)

            # test scores
            new_score = pd.Series([classifier.score(emb_test_df, y_test)])
            new_test_score_row = {
                "test_score": new_score.iloc[0],
                "itr": iter
            }

            test_scores_df = pd.concat([test_scores_df, pd.DataFrame([new_test_score_row])], ignore_index=True)
            test_scores[model_name].append(test_scores_df)

            # Convergence
            if hasattr(classifier, 'converged_'):
                convergence_value = classifier.converged_
            else:
                convergence_value = 'None'
            # convergence_df["itr"] = iter
            new_convergence_row = {
                "convergence" : convergence_value,
                "itr": iter
            }
            convergence_df = pd.concat([convergence_df,pd.DataFrame([new_convergence_row])], ignore_index=True)
            convergence[model_name].append(convergence_df)


    # write the raw results in csv files
    utils.write_output(results, output_results_dir, output_filename_prefix, "output",)
    utils.write_output(validation_scores, output_results_dir, output_filename_prefix, "validation_scores")
    utils.write_output(test_scores, output_results_dir, output_filename_prefix, "test_scores")
    utils.write_output(convergence, output_results_dir, output_filename_prefix, "convergence")
    # if feature importance exists:
    # if len(feature_importance) > 0:
    #     utils.write_output(feature_importance, output_results_dir, output_filename_prefix, "feature_imp")

    # create plots for validation scores
    plot_validation_scores(validation_scores, os.path.join(output_dir, visualizations_dir, sub_dir), output_filename_prefix)


def get_standardized_datasets(train_df, test_df, label_col):
    drop_cols = [label_col]

    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df[label_col]

    X_test = test_df.drop(columns=drop_cols)
    y_test = test_df[label_col]

    # Standardize dataset
    min_max_scaler = MinMaxScaler()
    min_max_scaler_fit = min_max_scaler.fit(X_train)
    feature_names = min_max_scaler_fit.get_feature_names_out()

    # creating a pandas df with column headers = feature names so that the prediction model_params can also have the same order of feature names
    # this is needed while getting the feature_importances from the model_params and mapping it back to the feature names.
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
