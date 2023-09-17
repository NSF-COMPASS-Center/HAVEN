import os
import pandas as pd
from pathlib import Path
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch
import tqdm
from statistics import mean

from utils import utils, nn_utils, kmer_utils, visualization_utils
from utils.early_stopping import EarlyStopping
from prediction.models.nlp import fnn, cnn1d, rnn, lstm, transformer, kmer_fnn, transformer_model
from prediction.models.cv import cnn2d, cnn2d_pool


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
    training_settings = classification_settings["training_settings"]
    n_iters = classification_settings["n_iterations"]

    id_col = sequence_settings["id_col"]
    sequence_col = sequence_settings["sequence_col"]
    label_col = label_settings["label_col"]
    results = {}
    for iter in range(n_iters):
        print(f"Iteration {iter}")
        # 1. Read the data files
        df = utils.read_dataset(input_dir, input_file_names,
                                cols=[id_col, sequence_col, label_col])
        # 2. Transform labels
        df, index_label_map = utils.transform_labels(df, label_settings,
                                                           classification_type=classification_settings["type"])

        # only if the feature type is kmer, computer the kmer features over the entire datatset
        # TODO: can this be moved to nn_utils and computer kmer features only over the training dataset? will that affect the performance? how will we compute features for the testing dataset
        kmer_keys = None
        if sequence_settings["feature_type"] == "kmer":
            kmer_keys = kmer_utils.get_kmer_keys(df,
                                                 k=sequence_settings["kmer_settings"]["k"],
                                                 sequence_col=sequence_col)
            sequence_settings["kmer_keys"] = kmer_keys
        # 3. Split dataset
        # full df into training and testing datasets in the ratio configured in the config file
        train_df, test_df = utils.split_dataset(df, input_split_seeds[iter],
                                                classification_settings["train_proportion"], stratify_col=label_col)
        # split testing set into validation and testing datasets in equal proportion
        # so 80:20 will now be 80:10:10
        val_df, test_df = utils.split_dataset(test_df, input_split_seeds[iter], 0.5, stratify_col=label_col)
        train_dataset_loader = nn_utils.get_dataset_loader(train_df, sequence_settings, label_col)
        val_dataset_loader = nn_utils.get_dataset_loader(val_df, sequence_settings, label_col)
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

            if "kmer-fnn" in model_name:
                print(f"Executing K-mer-FNN in {mode} mode")
                model["input_dim"] = train_dataset_loader.dataset.get_kmer_keys_count()
                nlp_model = kmer_fnn.get_fnn_model(model)

            elif "fnn" in model_name:
                print(f"Executing FNN in {mode} mode")
                nlp_model = fnn.get_fnn_model(model)

            elif "cgr-cnn-pool" in model_name:
                print(f"Executing CGR-CNN-Pool in {mode} mode")
                model["img_size"] = sequence_settings["cgr_settings"]["img_size"]
                nlp_model = cnn2d_pool.get_cnn_model(model)

            elif "cgr-cnn" in model_name:
                print(f"Executing CGR-CNN in {mode} mode")
                model["img_size"] = sequence_settings["cgr_settings"]["img_size"]
                nlp_model = cnn2d.get_cnn_model(model)

            elif "cnn" in model_name:
                print(f"Executing CNN in {mode} mode")
                nlp_model = cnn1d.get_cnn_model(model)

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
            result_df, nlp_model = run_model(nlp_model, train_dataset_loader, val_dataset_loader, test_dataset_loader,
                                             model["loss"], training_settings, model_name, mode)
            #  Create the result dataframe and remap the class indices to original input labels
            result_df.rename(columns=index_label_map, inplace=True)
            result_df["y_true"] = result_df["y_true"].map(index_label_map)
            result_df["itr"] = iter
            results[model_name].append(result_df)
            torch.save(nlp_model.state_dict(), model_filepath.format(model_name=model_name, itr=iter))

    # write the raw results in csv files
    output_results_dir = os.path.join(output_dir, results_dir, sub_dir)
    utils.write_output(results, output_results_dir, output_prefix, "output")


def run_model(model, train_dataset_loader, val_dataset_loader, test_dataset_loader, loss, training_settings, model_name, mode):
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
                              lr_scheduler, early_stopper, tbw, model_name, e)
            # check if early stopping condition was satisfied and stop accordingly
            if early_stopper.early_stop:
                print("Breaking off training loop due to early stop")
                break

    result_df, _ = evaluate_model(model, test_dataset_loader, criterion, tbw, model_name, epoch=None, log_loss=False)
    return result_df, model


def run_epoch(model, train_dataset_loader, val_dataset_loader, criterion, optimizer, lr_scheduler, early_stopper, tbw, model_name,
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

    # Validation
    _, val_loss = evaluate_model(model, val_dataset_loader, criterion, tbw, model_name, epoch, log_loss=True)
    early_stopper(val_loss)
    return model


def evaluate_model(model, dataset_loader, criterion, tbw, model_name, epoch, log_loss=False):
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
                tbw.add_scalar(f"{model_name}/validation-loss", float(curr_val_loss), model.test_iter)
                pbar.set_description(
                    f"{model_name}/validation-loss = {float(curr_val_loss)}, model.n_iter={model.test_iter}, epoch={epoch + 1}")
            val_loss.append(curr_val_loss)
            # to get probabilities of the output
            output = F.softmax(output, dim=-1)
            result_df = pd.DataFrame(output.cpu().numpy())
            result_df["y_true"] = label.cpu().numpy()
            results.append(result_df)
    return pd.concat(results, ignore_index=True), mean(val_loss)
