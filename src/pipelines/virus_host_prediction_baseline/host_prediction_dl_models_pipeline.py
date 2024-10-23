import os
import pandas as pd
from pathlib import Path
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn.functional as F
import torch
import wandb

from utils import utils, dataset_utils, nn_utils, constants, model_map
from training.early_stopping import EarlyStopping
from training import training_utils


def execute(input_settings, output_settings, classification_settings):
    # input settings
    input_dir = input_settings["input_dir"]
    input_file_names = input_settings["file_names"]

    # output settings
    output_dir = output_settings["output_dir"]
    results_dir = output_settings["results_dir"]
    sub_dir = output_settings["sub_dir"]
    output_prefix = output_settings["prefix"]
    output_prefix = output_prefix if output_prefix is not None else ""

    models = classification_settings["models"]
    label_settings = classification_settings["label_settings"]
    sequence_settings = classification_settings["sequence_settings"]
    training_settings = classification_settings["training_settings"]
    n_iters = classification_settings["n_iterations"]

    id_col = sequence_settings["id_col"]
    sequence_col = sequence_settings["sequence_col"]
    label_col = label_settings["label_col"]
    results = {}

    wandb_config = {
        "n_epochs": training_settings["n_epochs"],
        "lr": training_settings["max_lr"],
        "max_sequence_length": sequence_settings["max_sequence_length"],
        "dataset": input_file_names[0],
        "output_prefix": output_prefix
    }

    # model_params store filepath
    model_store_filepath = os.path.join(output_dir, results_dir, sub_dir, "{output_prefix}_{model_name}_itr{itr}.pth")
    Path(os.path.dirname(model_store_filepath)).mkdir(parents=True, exist_ok=True)

    for iter in range(n_iters):
        print(f"Iteration {iter}")
        # 1. Read the data files
        df = dataset_utils.read_dataset(input_dir, input_file_names,
                                        cols=[id_col, sequence_col, label_col])
        # 2. Transform labels
        df, index_label_map = utils.transform_labels(df, label_settings,
                                                     classification_type=classification_settings["type"])

        train_dataset_loader = None
        val_dataset_loader = None
        test_dataset_loader = None
        # 3. Split dataset
        if classification_settings["split_input"]:
            input_split_seeds = input_settings["split_seeds"]
            # full df into training and testing datasets in the ratio configured in the config file
            train_df, test_df = dataset_utils.split_dataset_stratified(df, input_split_seeds[iter],
                                                                       classification_settings["train_proportion"],
                                                                       stratify_col=label_col)
            # split testing set into validation and testing datasets in equal proportion
            # so 80:20 will now be 80:10:10
            val_df, test_df = dataset_utils.split_dataset_stratified(test_df, input_split_seeds[iter], 0.5,
                                                                     stratify_col=label_col)
            train_dataset_loader = dataset_utils.get_dataset_loader(train_df, sequence_settings, label_col)
            val_dataset_loader = dataset_utils.get_dataset_loader(val_df, sequence_settings, label_col)
            test_dataset_loader = dataset_utils.get_dataset_loader(test_df, sequence_settings, label_col)
        else:
            # used in zero shot evaluation, where split_input=False in classification_settings and mode=test in model_params
            test_dataset_loader = dataset_utils.get_dataset_loader(df, sequence_settings, label_col)

        dl_model = None
        for model in models:
            model_name = model["name"]
            # Set necessary values within model_params object for cleaner code and to avoid passing multiple arguments.
            model["vocab_size"] = constants.VOCAB_SIZE
            mode = model["mode"]

            if model["active"] is False:
                print(f"Skipping {model_name} ...")
                continue

            if model_name in model_map.model_map:
                print(f"Executing {model_name} in {mode} mode.")
                dl_model = model_map.model_map[model_name].get_model(model_params=model)
            else:
                print(f"ERROR: Unknown model {model_name}.")
                continue

            if model_name not in results:
                # first iteration
                results[model_name] = []

            # Initialize Weights & Biases for each run
            wandb_config["hidden_dim"] = model["hidden_dim"]
            wandb.init(project="zoonosis-host-prediction",
                       config=wandb_config,
                       group=classification_settings["experiment"],
                       job_type=model_name,
                       name=f"iter_{iter}")

            if mode == "train":
                # train the model_params
                result_df, dl_model = run_model(dl_model, train_dataset_loader, val_dataset_loader, test_dataset_loader,
                                                 model["loss"], training_settings, model_name)
            elif mode == "test":
                # used for zero-shot evaluation
                # load the pre-trained model_params
                dl_model.load_state_dict(torch.load(model["pretrained_model_path"]))
                result_df = test_model(dl_model, test_dataset_loader)
            else:
                print(f"ERROR: Unsupported mode '{mode}'. Supported values: 'train', 'test'.")
                exit(1)

            #  Create the result dataframe and remap the class indices to original input labels
            result_df.rename(columns=index_label_map, inplace=True)
            result_df["y_true"] = result_df["y_true"].map(index_label_map)
            result_df["itr"] = iter
            results[model_name].append(result_df)

            if classification_settings["save_model"]:
                # save the trained model_params
                model_filepath = model_store_filepath.format(output_prefix=output_prefix, model_name=model_name, itr=iter)
                torch.save(dl_model.state_dict(), model_filepath)
                print(f"Model output written to {model_filepath}")

            wandb.finish()
    # write the raw results in csv files
    output_results_dir = os.path.join(output_dir, results_dir, sub_dir)
    utils.write_output(results, output_results_dir, output_prefix, "output")


def run_model(model, train_dataset_loader, val_dataset_loader, test_dataset_loader, loss, training_settings, model_name):
    class_weights = utils.get_class_weights(train_dataset_loader).to(nn_utils.get_device())
    criterion = nn_utils.get_criterion(loss, class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    n_epochs = training_settings["n_epochs"]
    lr_scheduler = OneCycleLR(
        optimizer=optimizer,
        max_lr=float(training_settings["max_lr"]),
        epochs=n_epochs,
        steps_per_epoch=len(train_dataset_loader),
        pct_start=training_settings["pct_start"],
        anneal_strategy="cos",
        div_factor=training_settings["div_factor"],
        final_div_factor=training_settings["final_div_factor"])
    early_stopper = EarlyStopping(patience=10, min_delta=0)
    model.train_iter = 0
    model.val_iter = 0

    # START: Model training with early stopping using validation
    for e in range(n_epochs):
        model = training_utils.run_epoch(model, train_dataset_loader, val_dataset_loader, criterion, optimizer,
                                         lr_scheduler, early_stopper, model_name, e)
        # check if early stopping condition was satisfied and stop accordingly
        if early_stopper.early_stop:
            print("Breaking off training loop due to early stop")
            break
    # END: Model training with early stopping using validation

    # choose the model_params with the lowest validation loss from the early stopper
    best_performing_model = early_stopper.get_current_best_model()

    # test the model_params
    result_df = training_utils.test_model(best_performing_model, test_dataset_loader)

    return result_df, best_performing_model