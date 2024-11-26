import os
import pandas as pd
from pathlib import Path
import torch.nn.functional as F
import torch
import tqdm
from statistics import mean

from utils import utils, dataset_utils, nn_utils, constants, mapper
from models.baseline.nlp.transformer.transformer import TransformerEncoder
from few_shot_learning.prototypical_network_few_shot_classifier import PrototypicalNetworkFewShotClassifier

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

    pre_train_encoder_settings = pre_train_settings["encoder_settings"]
    pre_train_encoder_settings["vocab_size"] = constants.VOCAB_SIZE

    sequence_settings["max_sequence_length"] = pre_train_encoder_settings["max_seq_len"]

    tasks = fine_tune_settings["task_settings"]
    label_settings = fine_tune_settings["label_settings"]
    id_col = sequence_settings["id_col"]
    sequence_col = sequence_settings["sequence_col"]
    label_col = label_settings["label_col"]
    results = {}
    df = dataset_utils.read_dataset(input_dir, input_file_names, cols=[id_col, sequence_col, label_col])
    if fine_tune_settings["split_input"]:
        # Case: Host prediction using fine-tuning where there is dataset split and label groupings
        iter = 0 # choose only one iteration (first)
        # 1. Transform labels
        df, idx_label_map = utils.transform_labels(df, label_settings,
                                                     classification_type=fine_tune_settings["classification_type"])


        test_dataset_loader = None
        # 2. Split dataset
        # full df into training and testing datasets in the ratio configured in the config file
        train_df, test_df = dataset_utils.split_dataset_stratified(df, input_settings["split_seeds"][iter],
                                                                   fine_tune_settings["train_proportion"],
                                                                   stratify_col=label_col)
        # split testing set into validation and testing datasets in equal proportion
        # so 80:20 will now be 80:10:10
        val_df, test_df = dataset_utils.split_dataset_stratified(test_df, input_split_seeds[iter], 0.5,
                                                                 stratify_col=label_col)
    else:
        # Case: Few shot learning where there is no dataset split and no label groupings
        label_idx_map, idx_label_map = utils.get_label_vocabulary(test_df[label_col].unique())
        print(f"label_idx_map={label_idx_map}\nidx_label_map={idx_label_map}")
        df[label_col] = df[label_col].transform(lambda x: label_idx_map[x] if x in label_idx_map else 0)
        test_df = df

    test_dataset_loader = dataset_utils.get_dataset_loader(test_df, sequence_settings, label_col, include_id_col=True)
    fine_tune_model = None
    for task in tasks:
        task_id = task["id"] # unique identifier
        task_name = task["name"]

        if task["active"] is False:
            print(f"Skipping {task_name} ...")
            continue

        # load pre-trained encoder model_params
        pre_trained_encoder_model = TransformerEncoder.get_transformer_encoder(pre_train_encoder_settings, task["cls_token"])
        task["pre_trained_model"] = pre_trained_encoder_model

        # add maximum sequence length of pretrained model_params as the segment size from the sequence_settings
        # in pre_train_encoder_settings it has been incremented by 1 to account for CLS token
        task["segment_len"] = sequence_settings["max_sequence_length"]

        if task_name in mapper.model_map:
            print(f"Executing {task_name}.")
            fine_tune_model = mapper.model_map[task_name].get_model(model_params=task)
        else:
            print(f"ERROR: Unknown model {task_name}.")
            continue

        if task["few_shot_classifier"]:

            few_shot_classifier = PrototypicalNetworkFewShotClassifier(pre_trained_model=fine_tune_model)
            # load the pre-trained few-shot classifier
            few_shot_classifier.load_state_dict(torch.load(task["fine_tuned_model_path"], map_location=nn_utils.get_device()))
            fine_tune_model = few_shot_classifier.pre_trained_model
        else:
            fine_tune_model.load_state_dict(torch.load(task["fine_tuned_model_path"], map_location=nn_utils.get_device()))
        embedding_df = evaluate_model(fine_tune_model, test_dataset_loader, id_col)

        #  remap the class indices to original input labels
        embedding_df["y_true"] = embedding_df["y_true"].map(idx_label_map)
        results[task_id] = [embedding_df]

    # write the raw results in csv files
    output_results_dir = os.path.join(output_dir, results_dir, sub_dir)
    utils.write_output(results, output_results_dir, output_prefix, "output")


def evaluate_model(model, dataset_loader, id_col):
    with torch.no_grad():
        model.eval()
        embeddings = []
        for _, record in enumerate(pbar := tqdm.tqdm(dataset_loader)):
            id, input, label = record
            _, embedding = model(input, embedding_only=True)
            embedding_df = pd.DataFrame(embedding.cpu().numpy())
            embedding_df[id_col] = id
            embedding_df["y_true"] = label.cpu().numpy()
            embeddings.append(embedding_df)
    return pd.concat(embeddings, ignore_index=True)