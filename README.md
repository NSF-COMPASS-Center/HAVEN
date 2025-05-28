# HAVEN: Hierarchical Attention for Viral protEin-based host iNference

![HAVEN](figures/virus_host_prediction.png)
## Overview
HAVEN is a viral protein language model pretrained on 1.2 million protein sequences belonging to _Viridae_ (viruses) from [UniRef90](https://www.uniprot.org/uniref?query=%28taxonomy_id%3A10239%29&facets=identity%3A0.9). 
This pretrained model can be fine-tuned to predict any properties for viral protein sequences. This repository includes code and examples of fine-tuning HAVEN for virus-host prediction.
Specifically, given a protein sequence of a virus, predict the host from which the sequence was sampled.

This repository also includes the implementation of [Prototypical Networks](https://proceedings.neurips.cc/paper_files/paper/2017/hash/cb8da6767461f2812ae4290eac7cbc42-Abstract.html) to predict hosts with few labeled samples.
This few-shot learning framework is used to generalize HAVEN to predict rare and unseen hosts, and hosts of unseen viruses. For more details refer the manuscript at **<insert-bioRxiv-link>**.

---
## Installation
### Code and Dependencies
- Clone this GitHub repository at the desired location.
- Setup [Conda](https://docs.conda.io/en/latest/) environment and install the dependencies.
    ```shell
    bash
    conda create -n haven python=3.11.8
    conda activate haven
    pip install -r requirements.txt
    ```
  The installation takes a few minutes to complete.
- [Install pytorch](https://pytorch.org/get-started/locally/) based on your GPU/CPU configuration.


### Weights & Biases
1. Create an account in [Weights & Biases](https://wandb.ai/site/).
2. Create a new project in Weights and Biases named `haven`.
3. Setup the `wandb` library by completing [Step 1 in the Quickstart](https://wandb.ai/quickstart?utm_source=app-resource-center&utm_medium=app&utm_term=quickstart).
    - Note: Do not forget to log in to Weights and Biases (`wandb login`) in the server where you intend to execute the experiment.
---
## Usage
All experiments involving HAVEN can be configured using configuration files (.yml or .yaml) and executed as follows: 
```shell
python src/run.py -c <path-to-config-file>
```
Example
```shell
python src/run.py -c input/config-files/virus_host_prediction/uniref90/fine-tuning-HAVEN.yaml
```

In this work, the different functionalities are implemented as "pipelines". Each configuration file contains information required to execute a pipeline. The `config_type` parameter in the configuration file is used to identify the pipeline that is to be executed.  


### Examples
The different pipelines used to pretrain and fine-tune HAVEN and their corresponding configuration files are:

| Pipeline                                             | Config Type                | Example config                                                                                                                                                                                         |
|:-----------------------------------------------------|:---------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Pretraining HAVEN using Masked Language Modeling     | masked_language_modeling   | [uniref90-mlm-msl256.yaml](input/config-files/transfer_learning/masked_language_modeling/uniref90-mlm-msl256.yaml)                                                                                     |
| Virus Host Prediction - train & test - HAVEN         | virus_host_prediction      | [uniref90-fine-tuning-host-prediction-multi.yaml](input/config-files/virus_host_prediction/uniref90/fine-tuning-virprobert.yaml)                                                                       |
| Virus Host Prediction - test only - HAVEN            | virus_host_prediction_test | [cov-s-host-prediction-multi-uniref90-sarscov2-variants.yaml](input/config-files/interpretability/sarscov2_variants/cov-s-host-prediction-multi-uniref90-sarscov2-variants.yaml)                       |
| Few Shot Learning  to predict unseen and rare hosts  | few_shot_learning          | [uniref90-fine-tuning-host-prediction-non-idv-multi-few-shot-learning.yaml](input/config-files/few_shot_learning/novel_host/uniref90-fine-tuning-host-prediction-non-idv-multi-few-shot-learning.yaml) |
| Few Shot Learning to predict hosts of unseen viruses | few_shot_learning          | [uniref90-fine-tuning-host-prediction-idv-multi-few-shot-learning.yaml](input/config-files/few_shot_learning/novel_virus/uniref90-fine-tuning-host-prediction-idv-multi-few-shot-learning.yaml)        |
| Generate embeddings of protein sequences using HAVEN | embedding_generation       | [uniref90-fine-tuning-host-prediction-multi-embedding.yaml](input/config-files/interpretability/embedding/uniref90-fine-tuning-host-prediction-multi-embedding.yaml)                                   |

---
### Config File Parameters
#### Common parameters used in all configuration types

#### Masked Language Modeling (`masked_language_modeling`)
#### Fine-tuning HAVEN for Virus Host Prediction (`virus_host_prediction`)
1. Configure a suitable experiment name to be used to reference the execution using the `experiment` parameter in the config.
2. Set the relative path to the input file(s) within `input_settings` using `input_dir` and `file_names` parameters.
3. Set the following sequence related parameters in `sequence_settings` with respect to the input data file -
   1. `id_col`: identifier column name
   2. `sequence_col`: Sequence column name
   3. `truncate`: Boolean (default: False) - should the sequence be truncated with respect to the configured maximum sequence length?
   4. `split_sequence`: Boolean (default: False) - should the sequence be *explicitly* segmented with respect to the configured maximum sequence length?
4. Set the path to the transformer-encoder pre-trained using masked language modeling and all related parameters in `pre_train_settings`
    1. Configure the transformer-encoder related parameters in `pre_train_settings.encoder_settings`
5. Configure the parameters related to fine-tuning the pre-trained model in `fine_tune_settings`
6. Configure the common names of the virus hosts species and other label related settings in `fine_tune_settings.label_settings`
7. The different types fine-tuning models can be configured in `task_settings` and each model can be activated using its `active` flag.
    1. `data_parallel`: Enable parallel execution on multiple GPUs if available.
8. Configure the output directory and the prefix to be used while naming the output file in `output_settings`
---
## Data
All datasets used to pretrain and fine-tune HAVEN are available in Zenodo. **<insert-zenodo-link-here>**

Information on the dataset preprocessing steps and script is available in the [Dataset Preprocessing Pipeline README](dataset_preprocessing_pipeline_README.md).

---
## Models
Download the pretrained and fine-tuned models of HAVEN from Zenodo. **<insert-zenodo-link-here>**

---
## License
HAVEN software and data are available for public use under the GNU GENERAL PUBLIC LICENSE v3. 

---
## Reference
If you use HAVEN in your research, please cite the following preprint:

**<insert-bioRxiv-link-here>**