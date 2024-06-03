import os
import pandas as pd
from pathlib import Path
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn
import torch
import tqdm
from statistics import mean
import wandb

from utils import utils, dataset_utils, nn_utils, evaluation_utils
from training.early_stopping import EarlyStopping
from models.nlp.transformer import transformer
from training import pre_training_masked_language_modeling


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

    training_settings = config["training_settings"]
    mlm_settings = config["mlm_settings"]
    encoder_settings = config["encoder_settings"]
    sequence_settings = config["sequence_settings"]

    n_iters = training_settings["n_iterations"]
    id_col = sequence_settings["id_col"]
    sequence_col = sequence_settings["sequence_col"]
    pad_token_val = sequence_settings["pad_token_val"]

    # add n_tokens, max_sequence_length to encoder_settings
    encoder_settings["n_tokens"] = mlm_settings["n_tokens"] + 2  # (accounting for pad_token_val and mask_token_val)
    encoder_settings["max_seq_len"] = sequence_settings["max_sequence_length"]

    # add pad_token_val, encoder_dim (which is defined in input_dim of encoder_settings) to mlm_settings
    mlm_settings["pad_token_val"] = pad_token_val
    mlm_settings["encoder_dim"] = encoder_settings["input_dim"]

    wandb_config = {
        "n_epochs": training_settings["n_epochs"],
        "lr": training_settings["max_lr"],
        "max_sequence_length": sequence_settings["max_sequence_length"],
        "batch_size": sequence_settings["batch_size"],
        "dataset": input_file_names[0]
    }

    # Path to store the pre-trained encoder model
    encoder_model_name = encoder_settings["model_name"]
    encoder_model_filepath = os.path.join(output_dir, results_dir, sub_dir, encoder_model_name + "_itr{itr}.pth")
    mlm_checkpoint_filepath = os.path.join(output_dir, results_dir, sub_dir, "checkpoints", encoder_model_name + "_itr{itr}_checkpt{checkpt}.pth")
    # creating parent directories for mlm_checkpoint_filepath ensures that all parent directories for encoder_model_filepath are also created.
    Path(os.path.dirname(mlm_checkpoint_filepath)).mkdir(parents=True, exist_ok=True)

    results = {}

    for iter in range(n_iters):
        print(f"Iteration {iter}")
        # Initialize Weights & Biases for each run
        wandb_config["hidden_dim"] = encoder_settings["hidden_dim"]
        wandb_config["depth"] = encoder_settings["depth"]
        wandb.init(project="zoonosis-host-prediction",
                   config=wandb_config,
                   group=training_settings["experiment"],
                   job_type=encoder_model_name,
                   name=f"iter_{iter}")

        # 1. Read the data files
        df = dataset_utils.read_dataset(input_dir, input_file_names,
                                cols=[id_col, sequence_col])

        # 2. Split dataset
        # full df into training and testing datasets in the ratio configured in the config file
        train_df, test_df = dataset_utils.split_dataset(df, input_split_seeds[iter],
                                                                   training_settings["train_proportion"])
        # split testing set into validation and testing datasets in equal proportion
        # so 80:20 will now be 80:10:10
        val_df, test_df = dataset_utils.split_dataset(test_df, input_split_seeds[iter], 0.5)

        train_dataset_loader = dataset_utils.get_dataset_loader(train_df, sequence_settings, exclude_label=True)
        val_dataset_loader = dataset_utils.get_dataset_loader(val_df, sequence_settings, exclude_label=True)
        test_dataset_loader = dataset_utils.get_dataset_loader(test_df, sequence_settings, exclude_label=True)

        # 3. instantiate the encoder model
        encoder_model = transformer.get_transformer_encoder(encoder_settings)

        # 4. instantiate the mlm model
        mlm_model = pre_training_masked_language_modeling.get_mlm_model(encoder_model=encoder_model,
                                                                        mlm_model=mlm_settings)

        mlm_model = run(mlm_model, train_dataset_loader, val_dataset_loader, test_dataset_loader,
                        training_settings, encoder_model_name, pad_token_val,
                        mlm_checkpoint_filepath.replace("{itr}", str(iter)))
        torch.save(mlm_model.encoder_model.state_dict(), encoder_model_filepath.format(itr=iter))
        wandb.finish()

def run(model, train_dataset_loader, val_dataset_loader, test_dataset_loader,
        training_settings, encoder_model_name, pad_token_val, mlm_checkpoint_filepath):
    tbw = SummaryWriter()
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_val)
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

    # check and resume from checkpoint, if available
    last_epoch = -1
    if training_settings["checkpoint_path"]:
        model, optimizer, lr_scheduler, last_epoch = nn_utils.load_checkpoint(model, optimizer, lr_scheduler,
                                                                  training_settings["checkpoint_path"])
    model.train_iter = 0
    model.test_iter = 0

    for e in range(last_epoch+1, n_epochs):
        model = run_epoch(model, train_dataset_loader, val_dataset_loader, criterion, optimizer,
                          lr_scheduler, early_stopper, tbw, encoder_model_name, e)
        # check if early stopping condition was satisfied and stop accordingly
        if early_stopper.early_stop:
            print("Breaking off training loop due to early stop")
            break

        # save checkpoint
        nn_utils.save_checkpoint(model.state_dict(),
                                 optimizer.state_dict(),
                                 lr_scheduler.state_dict(),
                                 e, mlm_checkpoint_filepath)

    evaluate_model(model, test_dataset_loader, criterion, tbw, encoder_model_name, epoch=None, log_loss=False)
    return model


def run_epoch(model, train_dataset_loader, val_dataset_loader, criterion, optimizer, lr_scheduler, early_stopper, tbw, encoder_model_name,
              epoch):
    # Training
    model.train()
    for _, input in enumerate(pbar := tqdm.tqdm(train_dataset_loader)):
        optimizer.zero_grad()

        output, label = model(input)
        # transpose from b x max_seq_len x n_tokens -> b x n_tokens x max_seq_len
        # because CrossEntropyLoss expected input to be of the shape b x n_classes x number_dimensions_for_loss
        # in this case, number_of_dimensions_for_loss = max_seq_len as every sequences in the batch will have a loss corresponding to each token position
        output = output.transpose(1, 2).to(nn_utils.get_device())
        loss = criterion(output, label.long())
        loss.backward()

        optimizer.step()
        lr_scheduler.step()

        model.train_iter += 1
        curr_lr = lr_scheduler.get_last_lr()[0]
        train_loss = loss.item()
        # log training loss
        wandb.log({
            "learning-rate": float(curr_lr),
            "training-loss": float(train_loss)
        })

        tbw.add_scalar(f"{encoder_model_name}/learning-rate", float(curr_lr), model.train_iter)
        tbw.add_scalar(f"{encoder_model_name}/training-loss", float(train_loss), model.train_iter)
        pbar.set_description(
            f"{encoder_model_name}/training-loss = {float(train_loss)}, model.n_iter={model.train_iter}, epoch={epoch + 1}")

    # Validation
    val_loss = evaluate_model(model, val_dataset_loader, criterion, tbw, encoder_model_name, epoch, log_loss=True)
    early_stopper(val_loss)
    return model


def evaluate_model(model, dataset_loader, criterion, tbw, encoder_model_name, epoch, log_loss=False):
    with torch.no_grad():
        model.eval()

        val_loss = []
        for _, input in enumerate(pbar := tqdm.tqdm(dataset_loader)):

            output, label = model(input)
            # transpose from b x max_seq_len x n_tokens -> b x n_tokens x max_seq_len
            # because CrossEntropyLoss expected input to be of the shape b x n_classes x number_dimensions_for_loss
            # in this case, number_dimensions_for_loss = max_seq_len as every sequences in the batch will have a loss corresponding to each token position
            output = output.transpose(1, 2).to(nn_utils.get_device())
            loss = criterion(output, label.long())

            curr_val_loss = loss.item()
            model.test_iter += 1
            if log_loss:
                f1_micro, f1_macro = evaluation_utils.get_f1_score(y_true=label, y_pred=output, select_non_zero=True)
                # log validation loss
                wandb.log({
                    "validation-loss": float(curr_val_loss),
                    "f1_micro": f1_micro,
                    "f1_macro": f1_macro
                })
                tbw.add_scalar(f"{encoder_model_name}/validation-loss", float(curr_val_loss), model.test_iter)
                tbw.add_scalar(f"{encoder_model_name}/f1_micro", f1_micro, model.test_iter)
                tbw.add_scalar(f"{encoder_model_name}/f1_macro", f1_macro, model.test_iter)
                pbar.set_description(
                    f"{encoder_model_name}/validation-loss = {float(curr_val_loss)}, model.n_iter={model.test_iter}, epoch={epoch + 1}")
            val_loss.append(curr_val_loss)
            # to get probabilities of the output
            output = F.softmax(output, dim=-1)

    return mean(val_loss)