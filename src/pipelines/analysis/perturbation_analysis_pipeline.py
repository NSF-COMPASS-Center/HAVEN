import os
from pathlib import Path
import torch

from utils import utils, dataset_utils, nn_utils, constants, mapper, training_utils, perturbation_analysis_utils
from models.baseline.nlp.transformer.transformer import TransformerEncoder


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
    model_id = None
    for model in models:
        model_id = model["id"]  # unique identifier
        model_name = model["name"]
        mode = model["mode"]

        if model["active"] is False:
            print(f"Skipping {model_name} ...")
            continue

        pre_train_encoder_settings = model["pre_train_settings"]
        pre_train_encoder_settings["vocab_size"] = constants.VOCAB_SIZE

        # load pre-trained encoder model_params
        pre_trained_encoder_model = TransformerEncoder.get_transformer_encoder(pre_train_encoder_settings,
                                                                               model["cls_token"])
        model["pre_trained_model"] = pre_trained_encoder_model
        model["segment_len"] = pre_train_encoder_settings["max_seq_len"]
        sequence_settings["max_sequence_length"] = pre_train_encoder_settings["max_seq_len"]

        if model_name in mapper.model_map:
            print(f"Executing {model_name} in {mode} mode.")
            prediction_model = mapper.model_map[model_name].get_model(model_params=model)
        else:
            print(f"ERROR: Unknown model {model_name}.")
            continue

        prediction_model.load_state_dict(torch.load(model["model_path"], map_location=nn_utils.get_device()))

    output_results_dir = os.path.join(output_dir, results_dir, sub_dir, model_id)
    # create any missing parent directories
    Path(output_results_dir).mkdir(parents=True, exist_ok=True)

    # already present output files
    preexisting_output_files = os.listdir(output_results_dir)

    print(f"Number of input files = {len(input_files)}")
    for input_file in input_files:
        # check if the input file has already been processed
        if perturbation_analysis_utils.is_input_file_processed(input_file, preexisting_output_files):
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
        result_df = training_utils.test_model_analysis(prediction_model, test_dataset_loader, id_col)

        # 5. Create the result dataframe and remap the class indices to original input labels
        result_df.rename(columns=index_label_map, inplace=True)
        result_df["y_true"] = result_df["y_true"].map(index_label_map)

        # 6. Write the raw results in csv files
        output_prefix_curr = output_prefix + "_" + Path(input_file).stem
        utils.write_analysis_output(result_df, output_results_dir, output_prefix_curr, output_type="output")

        # 7. clear memory
        del df, test_dataset_loader, result_df
