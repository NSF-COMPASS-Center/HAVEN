#!/usr/src/env python
import argparse
from utils import utils, mapper


def parse_args():
    parser = argparse.ArgumentParser(description="Zoonosis prediction pipeline")
    parser.add_argument('-c', '--config', required=True,
                        help="File containing configuration to execute the pipeline.\n")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    config = utils.parse_config(args.config)
    print(config)
    config_type = config["config_type"]
    if config_type in mapper.pipeline_mapper:
        mapper.pipeline_mapper[config_type].execute(config)
    else:
        print("ERROR: Unsupported configuration for config_type. See readme for supported 'config_type' values.")
        exit(1)
    return


if __name__ == "__main__":
    main()
    exit(0)
