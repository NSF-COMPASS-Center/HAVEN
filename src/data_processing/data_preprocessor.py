import os
import re


def execute(config):
    input_settings = config["input_settings"]
    input_dir = input_settings["input_dir"]
    input_dataset_dir = input_settings["dataset_dir"]
    input_files = input_settings["file_names"]

    output_settings = config["output_settings"]
    output_dir = output_settings["output_dir"]
    output_dataset_dir = output_settings["dataset_dir"]
    output_prefix = output_settings["prefix"]
    output_prefix = "_" + output_prefix if output_prefix is not None else ""

    labels = config["processor_settings"]["label_names"]

    for input_file in input_files:
        input_file_path = os.path.join(input_dir, input_dataset_dir, input_file)

        output_file_name = input_file + output_prefix + "_processed.csv"
        output_file_path = os.path.join(output_dir, output_dataset_dir, output_file_name)
        process_file(input_file_path, output_file_path, labels)


def process_file(input_file_path, output_file_path, labels):
    # label_capture_pattern = re.compile(">([a-zA-Z0-9. ]+)|([a-zA-Z0-9]+)|([a-zA-Z0-9]+)|([a-zA-Z0-9]+)", "")
    id = None
    region = None
    host = None
    genotype = None
    sequence = None
    lines_read = 0
    with open(input_file_path, "r") as input_file, open(output_file_path, "w+") as output_file:
        # write the header line in output file
        header_line = ["id", "region", *labels, "sequence"]
        output_file.write(",".join(header_line) + "\n")
        while True:
            line = input_file.readline().strip()
            if line is None or line == "":
                break
            lines_read += 1
            if line.startswith(">"):
                # if not the very first line, write the previously read record
                if lines_read > 1:
                    processed_line = ",".join([id, region, host, genotype, "".join(sequence)]) + "\n"
                    output_file.write(processed_line)

                # initialize sequence
                sequence = []

                # start of a new record
                match = re.search(r">(.+)\|(.+)\|(.+)\|(.+)", line)
                id = match.group(1).strip()
                region = match.group(2).strip()
                host = match.group(3).strip()
                genotype = match.group(4).strip()
            else:
                sequence.append(line)