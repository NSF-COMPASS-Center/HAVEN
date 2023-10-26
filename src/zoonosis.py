#!/usr/src/env python
import argparse
import yaml

from prediction import prediction
from evaluation import evaluation
from models.baseline import feature_importance
from utils import utils


def parse_args():
    parser = argparse.ArgumentParser(description="Zoonosis prediction pipeline")
    parser.add_argument('-c', '--config', required=True,
                        help="File containing configuration to execute the pipeline.\n")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    config = utils.parse_config(args.config)
    config_type = config["config_type"]
    if config_type == "prediction":
        prediction.execute(config)
    elif config_type == "evaluation":
        evaluation.execute(config)
    elif config_type == "feature_importance":
        feature_importance.execute(config)
    else:
        print("ERROR: Unsupported configuration for config_type. Supported values=data_preprocessor", "classification", "evaluation")
    return


if __name__ == '__main__':
    main()
    exit(0)
