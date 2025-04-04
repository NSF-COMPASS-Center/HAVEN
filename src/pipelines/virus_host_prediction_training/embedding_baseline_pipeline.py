import os
from pathlib import Path
from torch.optim.lr_scheduler import OneCycleLR
import torch
import tqdm
import wandb
import pandas as pd

from utils import utils, dataset_utils, nn_utils, constants, mapper, training_utils
from training_accessories.early_stopping import EarlyStopping
from models.baseline.nlp.transformer.transformer import TransformerEncoder

# from src.models.virprobert.virprobert_embeddings import VirProBERT_Emb


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

    pre_train_encoder_settings = pre_train_settings["encoder_settings"]
    pre_train_encoder_settings["vocab_size"] = constants.VOCAB_SIZE
    n_iters = fine_tune_settings["n_iterations"]

    sequence_settings["max_sequence_length"] = pre_train_encoder_settings["max_seq_len"]

    tasks = fine_tune_settings["task_settings"]
    id_col = sequence_settings["id_col"]
    sequence_col = sequence_settings["sequence_col"]
    label_col = label_settings["label_col"]
    results = {}

    wandb_config = {
        "n_epochs_freeze": training_settings["n_epochs_freeze"],
        "n_epochs_unfreeze": training_settings["n_epochs_unfreeze"],
        "lr": training_settings["max_lr"],
        "max_sequence_length": sequence_settings["max_sequence_length"],
        "dataset": input_file_names[0],
        "output_prefix": output_prefix
    }

    # fine_tune_model store filepath
    output_filepath = os.path.join(output_dir, results_dir, sub_dir, "{output_prefix}_{task_id}_itr{itr}.csv")
    Path(os.path.dirname(output_filepath)).mkdir(parents=True, exist_ok=True)

    for iter in range(n_iters):
        print(f"Iteration {iter}")
        # 1. Read the data files
        df = dataset_utils.read_dataset(input_dir, input_file_names,
                                cols=[id_col, sequence_col, label_col])
        # 2. Transform labels
        df, index_label_map = utils.transform_labels(df, label_settings,
                                                           classification_type=fine_tune_settings["classification_type"])

        train_dataset_loader = None
        val_dataset_loader = None
        test_dataset_loader = None
        # 3. Split dataset
        if fine_tune_settings["split_input"]:
            # full df into training and testing datasets in the ratio configured in the config file
            train_df, test_df = dataset_utils.split_dataset_stratified(df, input_settings["split_seeds"][iter],
                                                                       fine_tune_settings["train_proportion"], stratify_col=label_col)
            # split testing set into validation and testing datasets in equal proportion
            # so 80:20 will now be 80:10:10
            # val_df, test_df = dataset_utils.split_dataset_stratified(test_df, input_split_seeds[iter], 0.5, stratify_col=label_col)
            train_dataset_loader = dataset_utils.get_dataset_loader(train_df, sequence_settings, label_col)
            # val_dataset_loader = dataset_utils.get_dataset_loader(val_df, sequence_settings, label_col)
            test_dataset_loader = dataset_utils.get_dataset_loader(test_df, sequence_settings, label_col)
        else:
            # used in zero shot evaluation, where split_input=False in fine_tune_settings and mode=test in task
            test_dataset_loader = dataset_utils.get_dataset_loader(df, sequence_settings, label_col)

        embeddings = None
        for task in tasks:
            task_id = task["id"] # unique identifier
            task_name = task["name"]
            mode = task["mode"]

            if task["active"] is False:
                print(f"Skipping {task_name} ...")
                continue

            # load pre-trained encoder model_params
            pre_trained_encoder_model = TransformerEncoder.get_transformer_encoder(pre_train_encoder_settings, task["cls_token"])
            pre_trained_model_path = pre_train_settings["model_path"]
            if pre_trained_model_path:
                pre_trained_encoder_model.load_state_dict(
                    torch.load(pre_trained_model_path, map_location=nn_utils.get_device()))

            # HACK to load models from checkpoints. CAUTION: Use only under dire circumstances
            # pre_trained_encoder_model = nn_utils.load_model_from_checkpoint(pre_trained_encoder_model,
            #                                                                pre_train_settings["model_path"])

            # set the pre_trained model_params within the task config
            task["pre_trained_model"] = pre_trained_encoder_model

            # add maximum sequence length of pretrained model_params as the segment size from the sequence_settings
            # in pre_train_encoder_settings it has been incremented by 1 to account for CLS token
            task["segment_len"] = sequence_settings["max_sequence_length"]
            embeddings = []
            if mode == "train":
                if task_name in mapper.model_map:
                    print(f"Executing {task_name} in {mode} mode.")
                    model = mapper.model_map[task_name].get_model(model_params=task)
                    for _, record in enumerate(pbar := tqdm.tqdm(train_dataset_loader)):
                        input, label = record
                        # optimizer.zero_grad()
                        output = model.get_embedding(input)
                        output = output.to(nn_utils.get_device())
                        embeddings.append(output)
                        print("RETURNN EMBEDDINGS")
                    embeddings = pd.DataFrame(embeddings)
                    embeddings.to_csv(output_filepath)
                else:
                    print(f"ERROR: Unknown model {task_name}.")
                    continue
            elif mode == "test":
                if task_name in mapper.model_map:
                    print(f"Executing {task_name} in {mode} mode.")
                    embeddings = mapper.model_map[task_name].get_model(model_params=task,
                                                                  dataset_loader=test_dataset_loader)
                    for _, record in enumerate(pbar := tqdm.tqdm(test_dataset_loader)):
                        input, label = record
                        # optimizer.zero_grad()
                        output = model.get_embedding(input)
                        output = output.to(nn_utils.get_device())
                        embeddings.append(output)
                        print("RETURNN EMBEDDINGS")
                    embeddings = pd.DataFrame(embeddings)
                    embeddings.to_csv(output_filepath)
                else:
                    print(f"ERROR: Unknown model {task_name}.")
                    continue
            else:
                print(f"ERROR: Unsupported mode '{mode}'. Supported values: 'train', 'test'.")
                exit(1)
