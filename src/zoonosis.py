#!/usr/src/env python
import argparse

from pipelines.host_prediction import host_prediction_pipeline
from pipelines.transfer_learning import masked_language_modeling_pipleine, fine_tuning_host_prediction_pipeline
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
    # transfer-learning
    if config_type == "transfer_learning":
        config_sub_type = config_type["config_sub_type"]
        # transfer-learning: pre-training
        if config_sub_type == "masked_language_modeling":
            masked_language_modeling_pipleine.execute(config)
        # transfer-learning: fine-tuning
        elif config_sub_type == "host_prediction":
            fine_tuning_host_prediction_pipeline.execute(config)
        else:
            print("ERROR: Unsupported configuration for config_type. Supported values=masked_langage_modeling",
                  "host_prediction")
    # classification: host-prediction
    elif config_type == "host_prediction":
        host_prediction_pipeline.execute(config)
    # evaluation
    elif config_type == "evaluation":
        evaluation.execute(config)
    # feature_importance for baseline models
    # TODO: >>>> DEPRECATED <<<< (Remove all corresponding code)
    elif config_type == "feature_importance":
        feature_importance.execute(config)
    else:
        print("ERROR: Unsupported configuration for config_type. Supported values=transfer_learning", "host_prediction", "evaluation")
    return


if __name__ == '__main__':
    main()
    exit(0)
