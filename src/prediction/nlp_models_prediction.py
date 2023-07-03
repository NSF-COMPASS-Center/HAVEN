import os
import pandas as pd
from pathlib import Path
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch
import tqdm

from utils import utils, nn_utils, visualization_utils
from prediction.models.nlp import fnn, rnn, lstm, transformer


def execute(input_settings, output_settings, classification_settings):
    # input settings
    input_dir = input_settings["input_dir"]
    input_file_names = input_settings["file_names"]
    input_split_seeds = input_settings["split_seeds"]

    # output settings
    output_dir = output_settings["output_dir"]
    results_dir = output_settings["results_dir"]
    sub_dir = output_settings["sub_dir"]
    output_prefix = output_settings["prefix"]
    output_prefix = output_prefix if output_prefix is not None else ""

    models = classification_settings["models"]
    label_settings = classification_settings["label_settings"]
    sequence_settings = classification_settings["sequence_settings"]
    n_iters = classification_settings["n_iterations"]

    sequence_col = sequence_settings["sequence_col"]
    label_col = label_settings["label_col"]
    results = {}
    for iter in range(n_iters):
        print(f"Iteration {iter}")
        # 1. Read the data files
        df = utils.read_dataset(input_dir, input_file_names,
                                cols=[sequence_col, label_col])
        # 2. Transform labels
        df, index_label_map = utils.transform_labels(df, label_settings,
                                                           classification_type=classification_settings["type"])
        # 3. Split dataset
        train_df, test_df = utils.split_dataset(df, input_split_seeds[iter],
                                                classification_settings["train_proportion"], stratify_col=label_col)

        train_dataset_loader = nn_utils.get_dataset_loader(train_df, sequence_settings, label_col)
        test_dataset_loader = nn_utils.get_dataset_loader(test_df, sequence_settings, label_col)

        nlp_model = None
        # model store filepath
        model_filepath = os.path.join(output_dir, results_dir, sub_dir, "{model_name}_itr{itr}.pth")
        Path(os.path.dirname(model_filepath)).mkdir(parents=True, exist_ok=True)

        for model in models:
            model_name = model["name"]
            # Set necessary values within model object for cleaner code and to avoid passing multiple arguments.
            model["max_seq_len"] = sequence_settings["max_sequence_length"]
            mode = model["mode"]

            if model["active"] is False:
                print(f"Skipping {model_name} ...")
                continue

            if model_name not in results:
                # first iteration
                results[model_name] = []

            if "fnn" in model_name:
                print(f"Executing FNN in {mode} mode")
                nlp_model = fnn.get_fnn_model(model)

            elif "rnn" in model_name:
                print(f"Executing RNN in {mode} mode")
                nlp_model = rnn.get_rnn_model(model)

            elif "lstm" in model_name:
                print(f"Executing LSTM in {mode} mode")
                nlp_model = lstm.get_lstm_model(model)

            elif "transformer" in model_name:
                print(f"Executing Transformer in {mode} mode")
                nlp_model = transformer.get_transformer_model(model)

            else:
                continue

            # Execute the NLP model
            if mode == "test":
                nlp_model.load_state_dict(torch.load(model["pretrained_model_path"]))
            result_df, nlp_model = run_model(nlp_model, train_dataset_loader, test_dataset_loader,
                                             model["loss"],
                                             model["n_epochs"], model_name, mode)
            #  Create the result dataframe and remap the class indices to original input labels
            result_df.rename(columns=index_label_map, inplace=True)
            result_df["y_true"] = result_df["y_true"].map(index_label_map)
            result_df["itr"] = iter
            results[model_name].append(result_df)
            torch.save(nlp_model.state_dict(), model_filepath.format(model_name=model_name, itr=iter))

    # write the raw results in csv files
    output_results_dir = os.path.join(output_dir, results_dir, sub_dir)
    utils.write_output(results, output_results_dir, output_prefix, "output")


def run_model(model, train_dataset_loader, test_dataset_loader, loss, n_epochs, model_name, mode):
    tbw = SummaryWriter()
    class_weights = utils.get_class_weights(train_dataset_loader).to(nn_utils.get_device())
    criterion = nn_utils.get_criterion(loss, class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    lr_scheduler = OneCycleLR(
        optimizer=optimizer,
        max_lr=1e-4,
        epochs=n_epochs,
        steps_per_epoch=len(train_dataset_loader),
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=10000.0)
    model.train_iter = 0
    model.test_iter = 0
    if mode == "train":
        # train the model only if set to train mode
        for e in range(n_epochs):
            model = run_epoch(model, train_dataset_loader, test_dataset_loader, criterion, optimizer,
                              lr_scheduler, tbw, model_name, e)

    return evaluate_model(model, test_dataset_loader, criterion, tbw, model_name, epoch=None, log_loss=False), model


def run_epoch(model, train_dataset_loader, test_dataset_loader, criterion, optimizer, lr_scheduler, tbw, model_name,
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
        tbw.add_scalar(f"{model_name}/learning-rate", float(curr_lr), model.train_iter)
        tbw.add_scalar(f"{model_name}/training-loss", float(train_loss), model.train_iter)
        pbar.set_description(
            f"{model_name}/training-loss = {float(train_loss)}, model.n_iter={model.train_iter}, epoch={epoch + 1}")

    # Testing
    evaluate_model(model, test_dataset_loader, criterion, tbw, model_name, epoch, log_loss=True)
    return model


def evaluate_model(model, test_dataset_loader, criterion, tbw, model_name, epoch, log_loss=False):
    with torch.no_grad():
        model.eval()

        results = []
        for _, record in enumerate(pbar := tqdm.tqdm(test_dataset_loader)):
            input, label = record

            output = model(input)  # b x n_classes
            output = output.to(nn_utils.get_device())

            loss = criterion(output, label.long())
            val_loss = loss.item()
            model.test_iter += 1
            if log_loss:
                tbw.add_scalar(f"{model_name}/validation-loss", float(val_loss), model.test_iter)
                pbar.set_description(
                    f"{model_name}/validation-loss = {float(val_loss)}, model.n_iter={model.test_iter}, epoch={epoch + 1}")
            # to get probabilities of the output
            output = F.softmax(output, dim=-1)
            result_df = pd.DataFrame(output.cpu().numpy())
            result_df["y_true"] = label.cpu().numpy()
            results.append(result_df)
    return pd.concat(results, ignore_index=True)
