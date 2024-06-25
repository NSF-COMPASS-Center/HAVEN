import os
import pandas as pd
from pathlib import Path
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch
import tqdm

from utils import utils, dataset_utils, nn_utils
from models.nlp.transformer import transformer
from models.nlp import cnn1d, rnn, lstm, fnn
from models.cv import cnn2d, cnn2d_pool
from transfer_learning.fine_tuning import host_prediction


def execute(config):
    # input settings
    input_settings = config["input_settings"]
    input_dir = input_settings["perturbed_dataset_dir"]
    input_files = os.listdir(input_dir)

    # output settings
    output_settings = config["output_settings"]
    output_dir = output_settings["output_dir"]
    results_dir = output_settings["results_dir"]
    sub_dir = output_settings["sub_dir"]
    output_prefix = output_settings["prefix"]
    output_prefix = output_prefix if output_prefix is not None else ""

    classification_settings = config["classification_settings"]
    models = classification_settings["models"]
    label_settings = classification_settings["label_settings"]
    sequence_settings = classification_settings["sequence_settings"]

    id_col = sequence_settings["id_col"]
    sequence_col = sequence_settings["sequence_col"]
    label_col = label_settings["label_col"]

    prediction_model = None

    for model in models:
        model_name = model["name"]
        # Set necessary values within model object for cleaner code and to avoid passing multiple arguments.
        model["max_seq_len"] = sequence_settings["max_sequence_length"]
        mode = model["mode"]

        if model["active"] is False:
            print(f"Skipping {model_name} ...")
            continue

        if "transfer_learning" in model_name:
            print(f"Executing Transfer Learning (Pre-trained and fine tuned model) in {mode} mode")
            pre_train_encoder_settings = model["pre_train_settings"]
            pre_train_encoder_settings["n_tokens"] += 2
            # add max_sequence_length to pre_train_encoder_settings
            pre_train_encoder_settings["max_seq_len"] = sequence_settings["max_sequence_length"]
            # load pre-trained encoder model
            pre_trained_encoder_model = transformer.get_transformer_encoder(pre_train_encoder_settings)
            # Set the pre_trained model within the task config
            model["pre_trained_model"] = pre_trained_encoder_model
            prediction_model = host_prediction.get_host_prediction_model(model)

        elif "fnn" in model_name:
            print(f"Executing FNN in {mode} mode")
            prediction_model = fnn.get_fnn_model(model)

        elif "cgr-cnn-pool" in model_name:
            print(f"Executing CGR-CNN-Pool in {mode} mode")
            model["img_size"] = sequence_settings["cgr_settings"]["img_size"]
            prediction_model = cnn2d_pool.get_cnn_model(model)

        elif "cgr-cnn" in model_name:
            print(f"Executing CGR-CNN in {mode} mode")
            model["img_size"] = sequence_settings["cgr_settings"]["img_size"]
            prediction_model = cnn2d.get_cnn_model(model)

        elif "cnn" in model_name:
            print(f"Executing CNN in {mode} mode")
            prediction_model = cnn1d.get_cnn_model(model)

        elif "rnn" in model_name:
            print(f"Executing RNN in {mode} mode")
            prediction_model = rnn.get_rnn_model(model)

        elif "lstm" in model_name:
            print(f"Executing LSTM in {mode} mode")
            prediction_model = lstm.get_lstm_model(model)

        elif "host_prediction" in model_name:
            print(f"Executing Host Prediction fine tuning in {mode} mode")
            prediction_model = host_prediction.get_host_prediction_model(model)

        else:
                continue

        # Execute the NLP model
        if mode == "test":
            prediction_model.load_state_dict(torch.load(model["model_path"], map_location=nn_utils.get_device()))

    output_results_dir = os.path.join(output_dir, results_dir, sub_dir)
    # create any missing parent directories
    Path(output_results_dir).mkdir(parents=True, exist_ok=True)

    # already present output files
    preexisting_output_files = os.listdir(output_results_dir)

    print(f"Number of input files = {len(input_files)}")
    for input_file in input_files:
        # check if the input file has already been processed
        if is_input_file_processed(input_file, preexisting_output_files):
            print(f"Skipping preprocessed input: {input_file}")
            continue
        print(input_file)
        # 1. Read the input data file
        df = dataset_utils.read_dataset(input_dir, [input_file],
                                cols=[id_col, sequence_col, label_col])

        # 2. Transform labels
        df, index_label_map = utils.transform_labels(df, label_settings,
                                                     classification_type=classification_settings["type"], silent=True)

        # 3. Get dataset loader
        test_dataset_loader = dataset_utils.get_dataset_loader(df, sequence_settings, label_col, include_id_col=True)

        # 4. Generate predictions
        result_df = evaluate_model(prediction_model, test_dataset_loader, id_col)

        # 5. Create the result dataframe and remap the class indices to original input labels
        result_df.rename(columns=index_label_map, inplace=True)
        result_df["y_true"] = result_df["y_true"].map(index_label_map)

        # 6. Write the raw results in csv files
        output_prefix_curr = output_prefix + "_" + Path(input_file).stem
        write_output(result_df, output_results_dir, output_prefix_curr, output_type="output")

        # 7. clear memory
        del df, test_dataset_loader, result_df


def evaluate_model(model, dataset_loader, id_col):
    tbw = SummaryWriter()
    with torch.no_grad():
        model.eval()

        results = []
        val_loss = []
        for _, record in enumerate(pbar := tqdm.tqdm(dataset_loader)):
            id, input, label = record

            output = model(input)  # b x n_classes
            output = output.to(nn_utils.get_device())

            # to get probabilities of the output
            output = F.softmax(output, dim=-1)
            result_df = pd.DataFrame(output.cpu().numpy())
            result_df[id_col] = id
            result_df["y_true"] = label.cpu().numpy()
            results.append(result_df)
    return pd.concat(results, ignore_index=True)


def write_output(df, output_dir, output_prefix, output_type):
    output_file_name = f"{output_prefix}.csv"
    output_file_path = os.path.join(output_dir, output_file_name)
    # 5. Write the classification output
    print(f"Writing {output_type} to {output_file_path}: {df.shape}")
    df.to_csv(output_file_path, index=False)


def is_input_file_processed(input_file, preexisting_output_files):
    is_present = False

    for f in preexisting_output_files:
        if input_file in f:
            is_present = True
            break

    return is_present