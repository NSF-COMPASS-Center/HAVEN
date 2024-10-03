import os
import pandas as pd
from pathlib import Path
from torch.optim.lr_scheduler import OneCycleLR
import torch
import torch.nn as nn
import torch.nn.functional as F
from statistics import mean
import tqdm
import wandb

from models.nlp.transformer import transformer
from transfer_learning.fine_tuning import host_prediction
from models.nlp import cnn1d, rnn, lstm, fnn
from models.nlp.hybrid import transformer_attention
from training.early_stopping import EarlyStopping
from utils import utils, dataset_utils, nn_utils, evaluation_utils, constants
from few_shot_learning.prototypical_network_few_shot_classifier import PrototypicalNetworkFewShotClassifier


def execute(config):
    # input settings
    input_settings = config["input_settings"]
    input_dir = input_settings["input_dir"]
    input_file_names = input_settings["file_names"]

    # output settings
    output_settings = config["output_settings"]
    output_dir = output_settings["output_dir"]
    results_dir = output_settings["results_dir"]
    sub_dir = output_settings["sub_dir"]
    output_prefix = output_settings["prefix"]
    output_prefix = output_prefix if output_prefix is not None else ""

    sequence_settings = config["sequence_settings"]
    label_settings = config["label_settings"]

    few_shot_learn_settings = config["few_shot_learn_settings"]
    meta_train_settings  = few_shot_learn_settings["meta_train_settings"]
    meta_validate_settings = few_shot_learn_settings["meta_validate_settings"]
    meta_test_settings = few_shot_learn_settings["meta_test_settings"]
    n_iters = few_shot_learn_settings["n_iterations"]

    id_col = sequence_settings["id_col"]
    sequence_col = sequence_settings["sequence_col"]
    max_sequence_length = sequence_settings["max_sequence_length"]
    label_col = label_settings["label_col"]

    wandb_config = {
        "n_epochs": few_shot_learn_settings["n_epochs"],
        "lr": few_shot_learn_settings["max_lr"],
        "dataset": input_file_names[0],
        "outout_prefix": output_prefix
    }

    # model store filepath
    model_store_filepath = os.path.join(output_dir, results_dir, sub_dir, "{output_prefix}_{model_name}_itr{itr}.pth")
    Path(os.path.dirname(model_store_filepath)).mkdir(parents=True, exist_ok=True)

    results = {}
    evaluation_metrics = {}

    for iter in range(n_iters):
        print(f"Iteration {iter}")
        # 1. Read the data files
        df = dataset_utils.read_dataset(input_dir, input_file_names,
                                        cols=[id_col, sequence_col, label_col])

        train_dataset_loader = None
        val_dataset_loader = None
        test_dataset_loader = None

        # 2. Split dataset
        if few_shot_learn_settings["split_input"]:
            input_split_seeds = input_settings["split_seeds"]
            train_df, val_df, test_df = dataset_utils.split_dataset_for_few_shot_learning(df, label_col=label_col,
                                                                                          train_proportion=few_shot_learn_settings["train_proportion"],
                                                                                          val_proportion=few_shot_learn_settings["val_proportion"],
                                                                                          test_proportion=few_shot_learn_settings["test_proportion"],
                                                                                          seed=input_split_seeds[iter])

            train_dataset_loader = dataset_utils.get_episodic_dataset_loader(train_df, sequence_settings, label_col, meta_train_settings)
            val_dataset_loader = dataset_utils.get_episodic_dataset_loader(val_df, sequence_settings, label_col, meta_validate_settings)
            test_dataset_loader = dataset_utils.get_episodic_dataset_loader(test_df, sequence_settings, label_col, meta_test_settings)

        prediction_model = None
        few_shot_classifier = None
        prediction_model_path = None
        prediction_models = config["pre_trained_models"]

        for model in prediction_models:
            model_name = model["name"]
            prediction_model_path = model["path"]
            mode = model["mode"]
            # when mode == 'train', pre_trained_model_path points to a pre-trained host prediction classifier
            # when mode = 'test', pre_trained_model_path points to a pre-trained few shot classifier

            # Set necessary values within model_settings object for cleaner code and to avoid passing multiple arguments.
            model_settings = model["model_settings"]

            if model["active"] is False:
                print(f"Skipping {model_name} ...")
                continue

            if model_name not in results:
                # first iteration
                results[model_name] = []
                evaluation_metrics[model_name] = []

            if "transformer" in model_name:
                print(f"Executing Transformer")
                prediction_model = transformer.get_transformer_encoder_classifier(model_settings)

            elif "virprobert" in model_name:
                print(f"Executing VirProBERT (pre-trained and fine tuned model).")
                # Load the pre-trained Transformer Encoder in the pre-trained (MLM) and fine-tuned (Host prediction) VirProBERT
                mlm_encoder_settings = model_settings["encoder_settings"].copy()
                mlm_encoder_settings["vocab_size"] = constants.VOCAB_SIZE
                # add max_sequence_length to pre_train_encoder_settings
                mlm_encoder_settings["max_seq_len"] = max_sequence_length
                # load pre-trained encoder model
                mlm_encoder_model = transformer.get_transformer_encoder(mlm_encoder_settings)

                # Set the pre_trained model within the virprobert model config
                # NOTE: this pre_trained_model is the MLM pre-trained model
                # this is different from the what we call "pre_trained" in the context of few-shot-learning,
                # i.e., model pre-trained for Host prediction.
                model_settings["pre_trained_model"] = mlm_encoder_model
                prediction_model = host_prediction.get_host_prediction_model(model_settings)

            elif "hybrid" in model_name:
                print(f"Executing Hybrid Attention Model (pre-trained and fine tuned model).")
                model_settings["encoder_settings"]["max_seq_len"] = sequence_settings["max_sequence_length"]
                mlm_encoder_settings = model_settings["encoder_settings"].copy()
                mlm_encoder_settings["vocab_size"] = constants.VOCAB_SIZE
                # add max_sequence_length to pre_train_encoder_settings
                mlm_encoder_settings["max_seq_len"] = max_sequence_length
                if model_settings["cls_token"]:
                    mlm_encoder_settings["max_seq_len"] += 1
                # load pre-trained encoder model
                mlm_encoder_model = transformer.get_transformer_encoder(mlm_encoder_settings)
                # Set the pre_trained model within the hybrid attn model config
                # NOTE: this pre_trained_model is the MLM pre-trained model
                # this is different from the what we call "pre_trained" in the context of few-shot-learning,
                # i.e., model pre-trained for Host prediction.
                model_settings["pre_trained_model"] = mlm_encoder_model
                # add maximum sequence length of pretrained model as the segment size from the sequence_settings
                # in pre_train_encoder_settings it has been incremented by 1 to account for CLS token, if needed
                model_settings["segment_len"] = max_sequence_length
                prediction_model = transformer_attention.get_model(model_settings)

            else:
                print(f"ERROR: Unrecognized model '{model_name}'.")
                continue

            # Initialize Weights & Biases for each run
            wandb.init(project="zoonosis-host-prediction",
                       config=wandb_config,
                       group=few_shot_learn_settings["experiment"],
                       job_type=model_name,
                       name=f"iter_{iter}")

            if mode == "train":
                # Load the pre-trained host prediction model
                prediction_model.load_state_dict(torch.load(prediction_model_path, map_location=nn_utils.get_device()))
                few_shot_classifier = PrototypicalNetworkFewShotClassifier(pre_trained_model=prediction_model)
                result_df, auprc_df, few_shot_classifier = run_few_shot_learning(few_shot_classifier, train_dataset_loader, val_dataset_loader, test_dataset_loader, few_shot_learn_settings,
                                      meta_train_settings, meta_validate_settings, meta_test_settings, model_name)
            elif mode == "test":
                # mode=test used for cross-domain few-shot evaluation: prediction of hosts in novel virus (hosts may or may not be novel)
                few_shot_classifier = PrototypicalNetworkFewShotClassifier(pre_trained_model=prediction_model)

                # load the pre-trained few-shot classifier
                few_shot_classifier.load_state_dict(torch.load(prediction_model_path, map_location=nn_utils.get_device()))
                result_df, auprc_df = meta_test_model(few_shot_classifier, test_dataset_loader, batch_size=few_shot_learn_settings["batch_size"])

            elif mode == "evaluate":
                meta_evaluate_settings = few_shot_learn_settings["meta_evaluate_settings"]
                # used in few shot evaluation, where split_input=False in classification_settings and mode=evaluate in model
                evaluate_dataset_loader = dataset_utils.get_evaluation_episodic_dataset_loader(sequence_settings,
                                                                                    label_col, meta_evaluate_settings)
                # mode=evaluate used for cross-domain few-shot evaluation: prediction of hosts in novel virus (hosts may or may not be novel)
                few_shot_classifier = PrototypicalNetworkFewShotClassifier(pre_trained_model=prediction_model)

                # load the pre-trained few-shot classifier
                few_shot_classifier.load_state_dict(torch.load(prediction_model_path, map_location=nn_utils.get_device()))
                result_df, auprc_df = meta_test_model(few_shot_classifier, evaluate_dataset_loader,
                                                      batch_size=few_shot_learn_settings["batch_size"])
            else:
                print(f"ERROR: Unsupported mode '{mode}'. Supported values are ['train', 'test', 'evaluate'].")
                exit(1)

            result_df["itr"] = iter
            auprc_df["itr"] = iter
            results[model_name].append(result_df)
            evaluation_metrics[model_name].append(auprc_df)

            if few_shot_learn_settings["save_model"]:
                # save the trained model
                model_filepath = model_store_filepath.format(output_prefix=output_prefix, model_name=model_name, itr=iter)
                torch.save(few_shot_classifier.state_dict(), model_filepath)
                print(f"Model output written to {model_filepath}")

            wandb.finish()

    # write the raw results in csv files
    output_results_dir = os.path.join(output_dir, results_dir, sub_dir)
    utils.write_output(results, output_results_dir, output_prefix, "output")
    utils.write_output(evaluation_metrics, output_results_dir, output_prefix, "classwise_auprc")


def run_few_shot_learning(model, train_dataset_loader, val_dataset_loader, test_dataset_loader,
                          few_shot_learning_settings, meta_train_settings, meta_validate_settings, meta_test_settings,
                          model_name):
    n_epochs = few_shot_learning_settings["n_epochs"]
    batch_size = few_shot_learning_settings["batch_size"]
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(few_shot_learning_settings["max_lr"]), weight_decay=1e-4)
    lr_scheduler = OneCycleLR(
        optimizer=optimizer,
        max_lr=float(few_shot_learning_settings["max_lr"]),
        epochs=n_epochs,
        steps_per_epoch=len(train_dataset_loader),
        pct_start=few_shot_learning_settings["pct_start"],
        anneal_strategy='cos',
        div_factor=few_shot_learning_settings["div_factor"],
        final_div_factor=few_shot_learning_settings["final_div_factor"])
    early_stopper = EarlyStopping(patience=10, min_delta=0)
    model.train_iter = 0
    model.val_iter = 0

    # meta training with validation
    for e in range(n_epochs):
        # training
        model = meta_train_model(model, train_dataset_loader, criterion, optimizer,
                          lr_scheduler, model_name, e, batch_size)
        # validation
        val_loss = meta_validate_model(model, val_dataset_loader, criterion, model_name, e, batch_size)
        early_stopper(model, val_loss)

        if early_stopper.early_stop:
            print("Breaking off training loop due to early stop.")
            break

    # choose the model with the lowest validation loss from the early stopper
    best_performing_model = early_stopper.get_current_best_model()

    # meta testing
    result_df, auprc_df = meta_test_model(best_performing_model, test_dataset_loader, batch_size)

    return result_df, auprc_df, best_performing_model


def meta_train_model(model, train_dataset_loader, criterion, optimizer, lr_scheduler, model_name, epoch, batch_size):
    model.train()
    for _, record in enumerate(pbar := tqdm.tqdm(train_dataset_loader)):
        support_sequences, support_labels, query_sequences, query_labels, _ = record

        optimizer.zero_grad()

        output = model(support_sequences, support_labels, query_sequences, batch_size)
        output = output.to(nn_utils.get_device())

        loss = criterion(output, query_labels.long())
        loss.backward()

        optimizer.step()
        lr_scheduler.step()

        model.train_iter += 1
        curr_lr = lr_scheduler.get_last_lr()[0]
        train_loss = loss.item()
        wandb.log({
            "learning-rate": float(curr_lr),
            "training-loss": float(train_loss)
        })
        pbar.set_description(
            f"{model_name}/training-loss = {float(train_loss)}, model.n_iter={model.train_iter}, epoch={epoch + 1}")
    return model


def meta_validate_model(model, val_dataset_loader, criterion, model_name, epoch, batch_size):
    with torch.no_grad():
        model.eval()

        val_loss = []
        for _, record in enumerate(pbar := tqdm.tqdm(val_dataset_loader)):
            support_sequences, support_labels, query_sequences, query_labels, _ = record

            output = model(support_sequences, support_labels, query_sequences, batch_size)
            output = output.to(nn_utils.get_device())

            loss = criterion(output, query_labels.long())
            curr_val_loss = loss.item()
            model.val_iter += 1

            # log validation loss
            wandb.log({
                "validation-loss": float(curr_val_loss)
            })
            pbar.set_description(
                f"{model_name}/validation-loss = {float(curr_val_loss)}, model.n_iter={model.val_iter}, epoch={epoch + 1}")
            val_loss.append(curr_val_loss)
    return mean(val_loss)


def meta_test_model(model, test_dataset_loader, batch_size):
    with torch.no_grad():
        model.eval()

        results = []
        evaluation_metrics = []
        for _, record in enumerate(pbar := tqdm.tqdm(test_dataset_loader)):
            support_sequences, support_labels, query_sequences, query_labels, idx_label_map = record

            output = model(support_sequences, support_labels, query_sequences, batch_size=batch_size)
            output = output.to(nn_utils.get_device())

            # to get probabilities of the output
            output = F.softmax(output, dim=-1)
            result_df = pd.DataFrame(output.cpu().numpy())
            result_df["y_true"] = query_labels.cpu().numpy()

            # remap the class indices to original input labels
            result_df.rename(columns=idx_label_map, inplace=True)
            result_df["y_true"] = result_df["y_true"].map(idx_label_map)
            results.append(result_df)

            _, auprc_df = evaluation_utils.compute_class_auprc(result_df, y_pred_columns=list(idx_label_map.values()), y_true_col="y_true")
            evaluation_metrics.append(auprc_df)

    return pd.concat(results, ignore_index=True), pd.concat(evaluation_metrics, ignore_index=True)