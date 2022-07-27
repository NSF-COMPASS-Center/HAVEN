#!/usr/src/env python
import argparse

import yaml

from hep import Hep
from hep_host import HepHost


def parse_args():
    parser = argparse.ArgumentParser(description='Hep sequence analysis pipeline')
    parser.add_argument('-c', '--config', required=True,
                        help="Configuration file containing list of datasets "
                             "algorithms and output specifications.\n")
    args = parser.parse_args()
    return args


# Returns a config map for the yaml at the path specified
def yaml_parse(path):
    try:
        with open(path, 'r') as f:
            return yaml.load(f, Loader=yaml.SafeLoader)
    except yaml.YAMLError as exc:
        print(f"Error parsing config file: {exc}")
        return None


def main():
    args = parse_args()
    args.config = yaml_parse(args.config)

    # extract respectively
    datasets = [v for d in args.config['input_settings']['datasets'].values() for k, v in d.items() if k == "path"]

    # Language model
    modelName = 'bilstm'
    # args.config['input_settings']['models'].keys()

    bilstmSettings = args.config['input_settings']['models'][modelName]
    if bilstmSettings['active']:
        print("Initalizing Bilstm Language Model----------------------------")
        h = Hep(model_name=modelName, datasets=datasets,
                n_epochs=bilstmSettings['epochs'],
                train=bilstmSettings['should_train'],
                checkpoint=bilstmSettings['checkpoint'],
                seed=bilstmSettings['seed'])
        h.start()

    datasets = [d for d in args.config['input_settings']['datasets'].values()]
    # Host model
    bilstmHostSettings = bilstmSettings[f'{modelName}_host']
    if bilstmHostSettings['active']:
        print("Initalizing Bilstm Host Model--------------------------------")
        hh = HepHost(model_name=modelName,
                     datasets=datasets,
                     targetNames=bilstmHostSettings['targetNames'],
                     targetKey=bilstmHostSettings['targetKey'],
                     train=bilstmHostSettings['should_train'],
                     n_epochs=bilstmHostSettings['epochs'],
                     n_hidden=bilstmHostSettings['n_hidden'],
                     embedding_dim=bilstmHostSettings['embedding_dim'],
                     train_split=bilstmHostSettings['train_split'],
                     batch_size=bilstmHostSettings['batch_size'],
                     inf_batch_size=bilstmHostSettings['inf_batch_size'],
                     test=bilstmHostSettings['should_test'],
                     transferCheckpoint=bilstmSettings['checkpoint'],
                     checkpoint=bilstmHostSettings['checkpoint'],
                     seed=bilstmSettings['seed'])

        hh.start()


if __name__ == '__main__':
    main()
