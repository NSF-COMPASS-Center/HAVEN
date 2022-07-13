#!/usr/bin/env python
import argparse
import yaml
from hep_host import HepHost
from hep import Hep

def parse_args():
    parser = argparse.ArgumentParser(description='Hep sequence analysis')
    parser.add_argument('-c','--config', required=True,
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


    print(args.config)
    '''
    def __init__(self, checkpoint:str=None, datasets:List[str]=None, transferCheckpoint:str=None,
            namespace:str="hep", model_name:str="bilstm",  seed:int=1, dim:int=512, batch_size:int=500, 
            n_epochs:int=11, train:bool=False, test:bool=False, embed:bool=False, 
            semantics:bool=False, combfit:bool=False, reinfection:bool=False):
    '''
 
    # extract respectively
    datasets = [v for d in args.config['input_settings']['datasets'] for k,v in d.items() if k == "path"]
    print(datasets)

    hh = HepHost(model_name="bilstm", datasets=datasets, 
            train=args.config['input_settings']['models']['bilstm_host']['should_train'],  
            transferCheckpoint=args.config['input_settings']['models']['bilstm']['checkpoint'], 
            seed=args.config['input_settings']['models']['bilstm_host']['seed'])
    hh.start()
    #h = Hep(model_name="bilstm", datasets=datasets, train=, 
    #        checkpoint=args.config['input_settings']['models']['bilstm']['checkpoint'],
    #        seed=args.config['input_settings']['models']['bilstm_host']['seed'])
    #h.start()

if __name__ == '__main__':
    main()
