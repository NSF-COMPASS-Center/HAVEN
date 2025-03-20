import subprocess
import os
from pathlib import Path

def execute(config):
    # input settings
    input_settings = config["input_settings"]
    input_dir = input_settings["input_dir"]
    input_file_name = input_settings["file_names"]
    alg_file_path = input_settings["alg_file_path"]

    # output settings
    output_settings = config["output_settings"]
    output_dir = output_settings["output_dir"]
    results_dir = output_settings["results_dir"]
    sub_dir = output_settings["sub_dir"]
    output_prefix = output_settings["prefix"]
    output_prefix = output_prefix if output_prefix is not None else ""

    # task settings
    ## TODO: improve this
    task_settings = config["task_settings"]
    id = task_settings[0]["id"]
    name = task_settings[0]["name"]
    sequence_type = task_settings[0]["sequence_type"]
    threshold = task_settings[0]["threshold"]
    word_size = task_settings[0]["word_size"]

    ## TODO: Make this a for loop to take in multiple input files (remove [0])
    # Path to store output file
    input_file = input_dir + "/" + input_file_name[0]
    input_file_path = os.path.join(input_file)
    output_file_name = (output_dir + "/" + results_dir + "/" +
                        input_file_name[0].replace(".fasta", "") + "_" + output_prefix)
    output_file_path = os.path.join(output_file_name)
    Path(os.path.dirname(output_file_path)).mkdir(parents=True, exist_ok=True)

    # Run the correct sequence clustering algorithm:
    if name == "CD-HIT":
        run_cd_hit(sequence_type, alg_file_path, input_file_path, output_file_path, threshold, word_size)
    elif name == "MMseqs2":
        run_mmseqs2()
    else:
        print("Incorrect name")
    # process_file(input_file_path, output_file_path, label_names)

def run_cd_hit(sequence_type, alg_file_path, input_file_path, output_file_path, threshold, word_size):
    if sequence_type == "Protein":
        cd_hit = alg_file_path + "/cd-hit"
        subprocess.run([cd_hit, "-i", input_file_path, "-o", output_file_path,
                        "-c", str(threshold), "-n", str(word_size)])
    elif sequence_type == "Nucleotide":
        cd_hit = alg_file_path + "/cd-hit-est"
        subprocess.run([cd_hit, "-i", input_file_path, "-o", output_file_path,
                        "-c", str(threshold), "-n", str(word_size)])
    else:
        print("Sequence type not supported")


def run_mmseqs2():
    pass

