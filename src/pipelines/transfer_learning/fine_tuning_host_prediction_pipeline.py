import os
import pandas as pd
from pathlib import Path
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch
import tqdm
from statistics import mean

from utils import utils, dataset_utils, nn_utils
from training.early_stopping import EarlyStopping
from training.fine_tuning import host_prediction


def execute(config):
    # input settings
    input_settings = config["input_settings"]
    input_dir = input_settings["input_dir"]
    input_file_names = input_settings["file_names"]
    input_split_seeds = input_settings["split_seeds"]

    # output settings
    output_settings = config["output_settings"]
    output_dir = output_settings["output_dir"]
    results_dir = output_settings["results_dir"]
    sub_dir = output_settings["sub_dir"]
    output_prefix = output_settings["prefix"]
    output_prefix = output_prefix if output_prefix is not None else ""

    sequence_settings = config["sequence_settings"]
    pre_train_settings = config["pre_train_settings"]

    fine_tune_settings = config["fine_tune_settings"]
    label_settings = fine_tune_settings["label_settings"]
    training_settings = fine_tune_settings["training_settings"]
    n_iters = fine_tune_settings["n_iterations"]

    tasks = config["task_settings"]

    id_col = sequence_settings["id_col"]
    sequence_col = sequence_settings["sequence_col"]
    label_col = label_settings["label_col"]
    results = {}

    for iter in range(n_iters):
        print(f"Iteration {iter}")
        # 1. Read the data files
        df = dataset_utils.read_dataset(input_dir, input_file_names,
                                cols=[id_col, sequence_col, label_col])
        # 2. Transform labels
        df, index_label_map = utils.transform_labels(df, label_settings,
                                                           classification_type=fine_tune_settings["classification_type"])

        # 3. Split dataset
        # full df into training and testing datasets in the ratio configured in the config file
        train_df, test_df = dataset_utils.split_dataset(df, input_split_seeds[iter],
                                                fine_tune_settings["train_proportion"], stratify_col=label_col)
        # split testing set into validation and testing datasets in equal proportion
        # so 80:20 will now be 80:10:10
        val_df, test_df = dataset_utils.split_dataset(test_df, input_split_seeds[iter], 0.5, stratify_col=label_col)
        train_dataset_loader = dataset_utils.get_dataset_loader(train_df, sequence_settings, label_col)
        val_dataset_loader = dataset_utils.get_dataset_loader(val_df, sequence_settings, label_col)
        test_dataset_loader = dataset_utils.get_dataset_loader(test_df, sequence_settings, label_col)

        # load pre-trained model
        pre_trained_model = None
        if "masked_language_model" in pre_train_settings["model_name"]:
            pre_trained_model.load_state_dict(torch.load(pre_train_settings["model_path"]))

        fine_tune_model = None

        # model store filepath
        fine_tune_model_filepath = os.path.join(output_dir, results_dir, sub_dir, "{task_name}_itr{itr}.pth")
        Path(os.path.dirname(model_filepath)).mkdir(parents=True, exist_ok=True)

        for task in tasks:
            task_name = task["name"]
            mode = model["mode"]
            # Set the pre_trained model within the task config
            task["pre_trained_model"] = pre_trained_model

            if task["active"] is False:
                print(f"Skipping {task_name} ...")
                continue

            if task_name not in results:
                # first iteration
                results[task_name] = []

            if "host_prediction" in task_name:
                print(f"Executing Host Prediction fine tuning in {mode} mode")
                fine_tune_model = host_prediction.get_host_prediction_model(task)
            else:
                continue

            # Execute the NLP model
            if mode == "test":
                fine_tune_model.load_state_dict(torch.load(task["fine_tuned_model_path"]))
            result_df, fine_tune_model = run_task(fine_tune_model, train_dataset_loader, val_dataset_loader, test_dataset_loader,
                                                   task["loss"], training_settings, task_name, mode)
            #  Create the result dataframe and remap the class indices to original input labels
            result_df.rename(columns=index_label_map, inplace=True)
            result_df["y_true"] = result_df["y_true"].map(index_label_map)
            result_df["itr"] = iter
            results[model_name].append(result_df)
            torch.save(nlp_model.state_dict(), model_filepath.format(model_name=model_name, itr=iter))

    # write the raw results in csv files
    output_results_dir = os.path.join(output_dir, results_dir, sub_dir)
    utils.write_output(results, output_results_dir, output_prefix, "output")


def run_task(model, train_dataset_loader, val_dataset_loader, test_dataset_loader, loss, training_settings, task_name, mode):
    tbw = SummaryWriter()
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
        anneal_strategy='cos',
        div_factor=training_settings["div_factor"],
        final_div_factor=training_settings["final_div_factor"])
    early_stopper = EarlyStopping(patience=10, min_delta=0)
    model.train_iter = 0
    model.test_iter = 0
    if mode == "train":
        # train the model only if set to train mode
        for e in range(n_epochs):
            model = run_epoch(model, train_dataset_loader, val_dataset_loader, criterion, optimizer,
                              lr_scheduler, early_stopper, tbw, task_name, e)
            # check if early stopping condition was satisfied and stop accordingly
            if early_stopper.early_stop:
                print("Breaking off training loop due to early stop")
                break

    result_df, _ = evaluate_model(model, test_dataset_loader, criterion, tbw, model_name, epoch=None, log_loss=False)
    return result_df, model


def run_epoch(model, train_dataset_loader, val_dataset_loader, criterion, optimizer, lr_scheduler, early_stopper, tbw, task_name,
              epoch):
    # Training
    model.train()
    for _, record in enumerate(pbar := tqdm.tqdm(train_dataset_loader)):
        input, label = record

        optimizer.zero_grad()

        output = model(input)
        output = output.to(nn_utils.get_device())

        loss = criterion(output, label.long())
        loss.backward()

        optimizer.step()
        lr_scheduler.step()

        model.train_iter += 1
        curr_lr = lr_scheduler.get_last_lr()[0]
        train_loss = loss.item()
        tbw.add_scalar(f"{task_name}/learning-rate", float(curr_lr), model.train_iter)
        tbw.add_scalar(f"{task_name}/training-loss", float(train_loss), model.train_iter)
        pbar.set_description(
            f"{model_name}/training-loss = {float(train_loss)}, model.n_iter={model.train_iter}, epoch={epoch + 1}")

    # Validation
    _, val_loss = evaluate_model(model, val_dataset_loader, criterion, tbw, task_name, epoch, log_loss=True)
    early_stopper(val_loss)
    return model


def evaluate_model(model, dataset_loader, criterion, tbw, task_name, epoch, log_loss=False):
    with torch.no_grad():
        model.eval()

        results = []
        val_loss = []
        for _, record in enumerate(pbar := tqdm.tqdm(dataset_loader)):
            input, label = record

            output = model(input)  # b x n_classes
            output = output.to(nn_utils.get_device())

            loss = criterion(output, label.long())
            curr_val_loss = loss.item()
            model.test_iter += 1
            if log_loss:
                tbw.add_scalar(f"{task_name}/validation-loss", float(curr_val_loss), model.test_iter)
                pbar.set_description(
                    f"{task_name}/validation-loss = {float(curr_val_loss)}, model.n_iter={model.test_iter}, epoch={epoch + 1}")
            val_loss.append(curr_val_loss)
            # to get probabilities of the output
            output = F.softmax(output, dim=-1)
            result_df = pd.DataFrame(output.cpu().numpy())
            result_df["y_true"] = label.cpu().numpy()
            results.append(result_df)
    return pd.concat(results, ignore_index=True), mean(val_loss)
