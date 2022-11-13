import os
import re
from pathlib import Path

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

    label_names = config["processor_settings"]["label_names"]

    for input_file in input_files:
        input_file_path = os.path.join(input_dir, input_dataset_dir, input_file)

        output_file_name = input_file + output_prefix + "_processed.csv"
        output_file_path = os.path.join(output_dir, output_dataset_dir, output_file_name)
        # create any missing parent directories
        Path(os.path.dirname(output_file_path)).mkdir(parents=True, exist_ok=True)
        process_file(input_file_path, output_file_path, label_names)


def process_file(input_file_path, output_file_path, label_names):
    id = None
    region = None
    host = None
    genotype = None
    sequence_record = None
    lines_read = 0
    with open(input_file_path, "r") as input_file, open(output_file_path, "w+") as output_file:
        # write the header line in output file
        header_line = ["id", "region", *label_names, "sequence"]
        output_file.write(",".join(header_line) + "\n")

        while True:
            line = input_file.readline().strip()
            if line is None or line == "":
                # EOF reached
                break
            lines_read += 1
            if line.startswith(">"):
                # '>' indicates the start of a new sequence record.
                if lines_read > 1:
                    # Unless the very first record, write the previously read sequence record
                    sequence_record_str = ",".join([id, region, host, genotype, "".join(sequence_record)]) + "\n"
                    output_file.write(sequence_record_str)

                # initialize a new sequence record
                sequence_record = []

                # parse the first line of a new sequence record of the form
                # >[id] | [region] |[host]|[genotype]

                # e.g. >QJQ50414.1 |ORF2|Avian|unknown
                match = re.search(r">(.+)\|(.+)\|(.+)\|(.+)", line)
                id = match.group(1).strip()
                region = match.group(2).strip()
                host = match.group(3).strip()
                genotype = match.group(4).strip()
            else:
                # if line does not begin with '>', it is a part of the protein sequence
                # append to the existing list of sequence record parts.
                sequence_record.append(line)
