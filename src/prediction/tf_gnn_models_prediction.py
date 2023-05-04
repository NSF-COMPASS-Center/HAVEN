import os
import pandas as pd
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch
import tqdm

from utils import utils, nn_utils, visualization_utils
from prediction.models.nlp import zoonoformer
from prediction.models.gnn import gnn_pipeline


def execute(input_settings, output_settings, classification_settings):
    # input settings
    seq_input_dir = input_settings["seq_input_dir"]
    inputs = input_settings["seq_file_names"]
    struct_input_dir = input_settings["struct_input_dir"]

    # output settings
    output_dir = output_settings["output_dir"]
    results_dir = output_settings["results_dir"]
    sub_dir = output_settings["sub_dir"]
    output_prefix = output_settings["prefix"]
    output_prefix = "_" + output_prefix if output_prefix is not None else ""

    models = classification_settings["models"]
    label_settings = classification_settings["label_settings"]
    sequence_settings = classification_settings["sequence_settings"]

    results = {}
    itr = 0
    _, ace2_graph = nn_utils.load_nx_graph(os.path.join(struct_input_dir, input_settings["ace2_prot_struct"]))
    for input in inputs:
        print(f"Iteration {itr}")
        # 1. Read the data files
        id_filepath = os.path.join(seq_input_dir, input["dir"], "{dataset_type}_ids.txt")
        seq_index_label_map, train_dataset_loader = nn_utils.get_protein_dataset_loader(id_filepath, struct_input_dir,
                                                                                    seq_input_dir, input,
                                                                                    sequence_settings,
                                                                                    label_settings,
                                                                                    dataset_type="train")
        seq_index_label_map, test_dataset_loader = nn_utils.get_protein_dataset_loader(id_filepath, struct_input_dir,
                                                                                   seq_input_dir, input,
                                                                                   sequence_settings,
                                                                                   label_settings, dataset_type="test")

        tf_model = None
        gnn_model = None

        for model in models:
            model_name = model["name"]
            if model["active"] is False:
                print(f"Skipping {model_name} ...")
                continue

            if model_name not in results:
                # first iteration
                results[model_name] = []

            if "tf-gnn" in model_name:
                # Set necessary values within model object for cleaner code and to avoid passing multiple arguments.
                model["max_seq_len"] = sequence_settings["max_sequence_length"]
                mode = model["mode"]
                print(f"Executing Transformer in {mode} mode")

                tf_model = zoonoformer.get_zoonoformer_model(model).to(nn_utils.get_device())
                gnn_model = gnn_pipeline.get_gnn_model(model).to(nn_utils.get_device())

                result_df = run_models(tf_model, gnn_model, ace2_graph, train_dataset_loader, test_dataset_loader, model["loss"],
                                       model["n_epochs"], model_name)
            else:
                continue

            #  Create the result dataframe and remap the class indices to original input labels
            result_df.rename(columns=seq_index_label_map, inplace=True)
            result_df["y_true"] = result_df["y_true"].map(seq_index_label_map)
            result_df["itr"] = itr
            results[model_name].append(result_df)
        itr += 1

    # write the raw results in csv files
    output_results_dir = os.path.join(output_dir, results_dir, sub_dir)
    utils.write_output(results, output_results_dir, output_prefix + "_", "output")


def run_models(tf_model, gnn_model, ace2_graph, train_dataset_loader, test_dataset_loader, loss, n_epochs, model_name):
    tbw = SummaryWriter()
    criterion = nn_utils.get_criterion(loss)
    tf_optimizer = torch.optim.Adam(tf_model.parameters(), lr=1e-4, weight_decay=1e-4)
    gnn_optimizer = torch.optim.Adam(tf_model.parameters(), lr=1e-4, weight_decay=1e-4)

    tf_lr_scheduler = OneCycleLR(
        optimizer=tf_optimizer,
        max_lr=1e-4,
        epochs=n_epochs,
        steps_per_epoch=len(train_dataset_loader),
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=10000.0)
    gnn_lr_scheduler = OneCycleLR(
        optimizer=gnn_optimizer,
        max_lr=1e-4,
        epochs=n_epochs,
        steps_per_epoch=len(train_dataset_loader),
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=10000.0)

    tf_model.train_iter = 0
    tf_model.test_iter = 0
    gnn_model.train_iter = 0
    gnn_model.test_iter = 0

    for e in range(n_epochs):
        tf_model, gnn_model = run_epoch(tf_model, gnn_model, ace2_graph, train_dataset_loader, test_dataset_loader, criterion, tf_optimizer, gnn_optimizer,
                          tf_lr_scheduler, gnn_lr_scheduler, tbw, model_name, e)

    return evaluate_model(tf_model, gnn_model, test_dataset_loader, ace2_graph, criterion, tbw, model_name, epoch=None, log_loss=False)


def run_epoch(tf_model, gnn_model, ace2_graph, train_dataset_loader, test_dataset_loader, criterion, tf_optimizer, gnn_optimizer,
                          tf_lr_scheduler, gnn_lr_scheduler, tbw, model_name,
              epoch):
    # Training
    tf_model.train()
    gnn_model.train()
    for _, record in enumerate(pbar := tqdm.tqdm(train_dataset_loader)):
        seq_input, seq_label, graph_input, graph_label = record

        tf_optimizer.zero_grad()
        gnn_optimizer.zero_grad()

        # train the gnn model
        graph_output, X_graph = gnn_model(ace2_graph, graph_input)
        graph_output = graph_output.to(nn_utils.get_device())

        # train the tf model
        seq_output = tf_model(seq_input, X_graph.to(nn_utils.get_device()))
        seq_output = seq_output.to(nn_utils.get_device())

        # gnn loss, gradient
        gnn_loss = criterion(graph_output, graph_label.long())
        gnn_loss.backward()

        gnn_optimizer.step()
        gnn_lr_scheduler.step()

        # tf loss, gradient
        tf_loss = criterion(seq_output, seq_label.long())
        tf_loss.backward()

        tf_optimizer.step()
        tf_lr_scheduler.step()

        tf_model.train_iter += 1
        gnn_model.train_iter += 1

        tf_curr_lr = tf_lr_scheduler.get_last_lr()[0]
        tf_train_loss = tf_loss.item()
        tbw.add_scalar(f"{model_name}-tf/learning-rate", float(tf_curr_lr), tf_model.train_iter)
        tbw.add_scalar(f"{model_name}-tf/training-loss", float(tf_train_loss), tf_model.train_iter)

        gnn_curr_lr = gnn_lr_scheduler.get_last_lr()[0]
        gnn_train_loss = gnn_loss.item()
        tbw.add_scalar(f"{model_name}-gnn/learning-rate", float(gnn_curr_lr), gnn_model.train_iter)
        tbw.add_scalar(f"{model_name}-gnn/training-loss", float(gnn_train_loss), gnn_model.train_iter)

        pbar.set_description(
            f"tf/train-loss = {float(tf_train_loss)}, tf_n_iter={tf_model.train_iter}, "
            f"gnn/train-loss = {float(gnn_train_loss)}, gnn_n_iter={gnn_model.train_iter}, "
            f"epoch={epoch + 1}")

    # Testing
    evaluate_model(tf_model, gnn_model, test_dataset_loader, ace2_graph, criterion, tbw, model_name, epoch, log_loss=True)
    return tf_model, gnn_model


def evaluate_model(tf_model, gnn_model, test_dataset_loader, ace2_graph, criterion, tbw, model_name, epoch, log_loss=False):
    with torch.no_grad():
        tf_model.eval()
        gnn_model.eval()

        results = []
        for _, record in enumerate(pbar := tqdm.tqdm(test_dataset_loader)):
            seq_input, seq_label, graph_input, graph_label = record

            # test the gnn model
            graph_output, X_graph = gnn_model(ace2_graph, graph_input)
            graph_output = graph_output.to(nn_utils.get_device())

            # test the tf model
            seq_output = tf_model(seq_input, X_graph.to(nn_utils.get_device()))
            seq_output = seq_output.to(nn_utils.get_device())

            # gnn loss
            gnn_loss = criterion(graph_output, graph_label.long())
            # tf loss
            tf_loss = criterion(seq_output, seq_label.long())

            tf_val_loss = tf_loss.item()
            gnn_val_loss = gnn_loss.item()
            tf_model.test_iter += 1
            gnn_model.test_iter += 1
            if log_loss:
                tbw.add_scalar(f"{model_name}-tf/val-loss", float(tf_val_loss), tf_model.test_iter)
                tbw.add_scalar(f"{model_name}-gnn/val-loss", float(gnn_val_loss), gnn_model.test_iter)
                pbar.set_description(
                    f"tf/val-loss = {float(tf_val_loss)},"
                    f"gnn/val-loss = {float(gnn_val_loss)}"
                    f"tf_n_iter={tf_model.test_iter},"
                    f"gnn_n_iter={gnn_model.test_iter},"
                    f"epoch={epoch + 1}")

            # to get probabilities of the output
            output = F.softmax(seq_output, dim=-1)
            result_df = pd.DataFrame(output.cpu().numpy())
            result_df["y_true"] = seq_label.cpu().numpy()
            results.append(result_df)
    return pd.concat(results, ignore_index=True)
