import pandas as pd
import os
from pathlib import Path
from evaluation.BinaryClassEvaluation import BinaryClassEvaluation
from evaluation.MultiClassEvaluation import MultiClassEvaluation


def execute(config):
    input_settings = config["input_settings"]
    input_dir = input_settings["input_dir"]
    input_dataset_dir = input_settings["dataset_dir"]
    input_file = input_settings["file_name"]

    output_settings = config["output_settings"]
    output_dir = output_settings["output_dir"]
    output_evaluation_dir = output_settings["evaluation_dir"]
    output_visualization_dir = output_settings["visualization_dir"]
    output_dataset_dir = output_settings["dataset_dir"]
    output_prefix = output_settings["prefix"]
    output_prefix = "_" + output_prefix if output_prefix is not None else ""

    input_file_path = os.path.join(input_dir, input_dataset_dir, input_file)
    print(f"input file = {input_file_path}")
    df = pd.read_csv(input_file_path)
    print(f"input results size = {df.shape}")

    output_file_name = Path(input_file_path).stem + "_" + output_prefix
    evaluation_output_file_base_path = os.path.join(output_dir, output_evaluation_dir, output_dataset_dir)
    visualization_output_file_base_path = os.path.join(output_dir, output_visualization_dir, output_dataset_dir)

    evaluation_settings = config["evaluation_settings"]
    evaluation_type = evaluation_settings["type"]
    if evaluation_type == "binary":
        evaluation_executor = BinaryClassEvaluation(df, evaluation_settings, evaluation_output_file_base_path, visualization_output_file_base_path, output_file_name)
    elif evaluation_type == "multi":
        evaluation_executor = MultiClassEvaluation(df, evaluation_settings, evaluation_output_file_base_path, visualization_output_file_base_path, output_file_name)
    else:
        print(f"ERROR: Unsupported type of evaluation: evaluation_settings.type = {evaluation_type}")

    evaluation_executor.execute()
    return
