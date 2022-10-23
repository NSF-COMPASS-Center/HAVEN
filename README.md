# About
Inspired by the UMAP projection clustering as pointed out in [Learning the language of viral evolution and escape](https://www.science.org/doi/10.1126/science.abd7331), language models seem to be able to cluster data very well during pretraining without supervision. We make a simple pipeline with yaml files that trains models and evaluates the predictive performance with and without pretraining.

# Prerequisites:
To run on ARC, you must copy this directory to ~/ on ARC and run deployment/setup1.sh and deployment/setup2.sh, which will set up a conda env labeled BioNLP.

### Data
Data is stored under ./data/hep in the form of Fasta files delimited by |, which contains protien, host, genotype in this order.
Data can be generated in GenbankData/ given you have an accession list.

### Dependencies
Install python dependencies via 
``` 
pip install -R ./requirements.txt
```

### Usage
General usage to call a yaml config and output to a log file.
```
python src/pipeline.py -c config/hepConfig.yaml > $RESULTS_DIR/hep_host_transfer.$(date +%Y_%b_%d_%H_%M).log 2>&1
```




