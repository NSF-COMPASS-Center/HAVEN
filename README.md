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

### 1. Configure your experiment
In this work, the different functionalities are implemented as "pipelines". For example, there are individual pipelines for pre-training HAVEN, fine-tuning HAVEN for virus-host prediction, and evaluating the performance of HAVEN using different datasets.
All the information needed to execute a particular experiment are configured in a "configuration file" (.yml or .yaml). Note that one configuration can support the execution of one pipeline only.

For more details on configuring experiments and examples, please refer the section on [Configuring Experiments](#configuring-experiments). Use the examples provided as reference to create your own configuration file.

### 2. Execute the configured experiment
Execute the following command
```shell
python src/run.py -c <path-to-config-file>
```
Example
```shell
python src/run.py -c input/config-files/virus_host_prediction/uniref90/fine-tuning-HAVEN.yaml
```

---
## Configuring Experiments
Each configuration file contains information required to execute a pipeline. 
Create a .yml or .yaml file in any location and use the below parameters to provide different information about the experiment being configured.

> [!NOTE]
> Each experiment will require the common parameters and some experiment-specific parameters

### Parameters common to all experiments / pipelines.
1. `config_type`: Unique identifier describing the experiment type that maps to one of the pipelines.
2. `input_settings`: All settings about input files.
   1. `input_dir`: Relative path to the folder where the input dataset files are located. The path is relative to the root 'HAVEN' folder.
   2. `file_names`: List of input file names to be read as the input dataset.
   3. `split_seeds`: List of random integers to be used as seeds while splitting the datasets.
3. `sequence_settings`: All settings about processing individual sequences in the dataset.
   1. `batch_size`: Batch size during training and testing.
   2. `id_col`: Name of the unique identifier column in the input dataset.
   3. `sequence_col`: Name of the column containing sequences in the input dataset.
   4. `max_sequence_length`: Length of individual segments. Each sequence is split into overlapping segments of this size.
   5. `split_sequence`: Boolean value indicating whether the sequences should be split into segments.
   6. `cls_token`: Boolean value indicating whether a 'CLS' token has to be appended at the beginning of each segment.
   7. `feature_type`: The type of features used to embed sequence. Supported values are ['token', 'kmer'].
4. `encoder_settings`: All settings about the 'Segment Encoder'. The 'Segment Encoder' architecture is  the encoder of [Transformer](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf).
   1. `model_name`: Unique name describing the model architecture. The value is used in naming the output files.
   2. `n_heads`: Number of heads in the multi-head self-attention of the transformer encoder.
   3. `depth`: Number of layers of the transformer encoder.
   4. `input_dim`: Dimension length of the embedding for each amino acid token when the token is converted into a vector.
   5. `hidden_dim`: Dimension length of the embedding for each amino acid token in all subsequent layers including the output layer.
5. `training_settings`: All settings about training the model.
   1. `experiment`: Unique name describing the experiment. The value is used in referencing this experiment in Weights and Biases.
   2. `n_iterations`: Number of times this experiment will be executed. This number should match the length of the list in `input_settings.split_seeds`. 
   3. `split_input`: Boolean value indicating whether the dataset has to be split into training and testing datasets. 
   If 'True', the values in `input_settings.split_seeds` are used as seeds sequentially in every iteration to randomly split the dataset while preserving the class proportions in the dataset.
   4. `train_proportion`: Value between 0 and 1 denoting the proportion of dataset to be used for training the model.
   5. `n_epochs`: Maximum number of epochs in one iteration of the experiment.
   6.  `max_lr`: Maximum learning rate used to train the model. [Annealing learning rate](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html) is used to train the models.
   7.  `pct_start`: Percentage of the epoch cycle (in number of steps or batches) spent increasing the learning rate. Refer [OneCycleLR](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html) for more details.
   8. `div_factor`: Factor used to determine the initial learning rate to begin increasing the learning rate. `initial_lr` = `max_lr`/`div_factor`. Refer [OneCycleLR](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html) for more details.
   9.  `final_div_factor`: Factor used to determine the minimum learning rate to end decreasing the learning rate. `min_lr` = `initial_lr`/`div_factor`. Refer [OneCycleLR](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html) for more details.
   10. `checkpoint_path`: Absolute path to a model checkpoint if you are resuming the training of HAVEN.
6. `output_settings`: All settings about output files.
   1. `output_dir`: Relative path to the folder where the output files to be written. The path is relative to the root 'HAVEN' folder.
   2. `results_dir`: Relative path to the subfolder where the output files to be written. The path is relative to `output_dir`.
   3. `prefix`: Value used to prefix all output files.

### Parameters specific to different experiments / pipelines
Each pipeline has some parameters are specific to its execution.

#### Masked Language Modeling
1. `mlm_settings`: All settings related to pre-training HAVEN using self-supervised masked language modeling.
   1. `mask_prob`: Probability with which a position in the sequence is selected for masking.
   2. `no_change_mask_prob`: Probability with which a selected position is left unchanged. 
   3. `random_mask_prob`: Probability with which a selected position is masked with a random token from the vocabulary of amino acids. All the remaining selected positions are masked with a 'MASK' token.

#### Virus Host Prediction
1. `pre_train_settings`: All settings related to the pre-trained model that is to be fine-tuned for virus-host prediction.
   1.  `model_name`: Name of the configuration or pipeline used for pre-training. For example "masked_language_modeling".
   2.  `model_path`: Absolute or relative path (with respect to HAVEN) where the pre-trained model file is located.
   3.  `encoder_settings`: Refer `encoder_settings` in [Parameters common to all experiments / pipelines](#parameters-common-to-all-experiments--pipelines).
2. `fine_tune_settings`: All settings related to pre-training HAVEN using self-supervised masked language modeling.
   1. `experiment`: Unique name describing the experiment. The value is used in referencing this experiment in Weights and Biases.
   2. `n_iterations`, `split_input`, `train_proportion`: Refer `training_settings` in [Parameters common to all experiments / pipelines](#parameters-common-to-all-experiments--pipelines).
   3. `classification_type`: The type of classification task for fine-tuning the mode. Supported values are ['binary', 'multi'].
   4. `save_model`: Boolean value indicating whether the final fine-tuned models have to be saved. If 'True', a model file is created for each iteration.
   5. `training_settings`: All settings related to fine-tuning the model for virus-host prediction.
      1. `n_epochs_freeze`: Maximum number of epochs for which the model weights of the pre-trained model are frozen. Fine-tuning begins with freezing the pre-trained models for this set number of epochs.
      2. `n_epochs_unfreeze`: After the model weights are released, maximum number of epochs for which the weights of the pre-trained model are updated.
      3. `max_lr`, `pct_start`, `div_factor`, `final_div_factor`: Refer `training_settings` in [Parameters common to all experiments / pipelines](#parameters-common-to-all-experiments--pipelines).
   6. `label_settings`: All settings related to grouping and / or renaming the labels.
      1. `label_col`:  Name of the column containing the label (class) for each sequence in the input dataset.
      2. `exclude_labels`: List of label values to be excluded from the dataset. All sequences containing any of the listed labels are removed from the input dataset.
      3. `label_groupings`: Mapping used to combine multiple labels into one group. Alternatively, this parameter can be used to rename the labels in the dataset at runtime.
   7. `task_settings`: All settings related to the models to 
### Example configuration files
The different pipelines used to pretrain and fine-tune HAVEN and their corresponding configuration files are:

| Pipeline                                             | Config Type                | Example config                                                                                                                                                                                         |
|:-----------------------------------------------------|:---------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Pretraining HAVEN using Masked Language Modeling     | masked_language_modeling   | [uniref90-mlm-msl256.yaml](input/config-files/transfer_learning/masked_language_modeling/uniref90-mlm-msl256.yaml)                                                                                     |
| Virus Host Prediction - train & test - HAVEN         | virus_host_prediction      | [uniref90-fine-tuning-host-prediction-multi.yaml](input/config-files/virus_host_prediction/uniref90/fine-tuning-haven.yaml)                                                                            |
| Virus Host Prediction - test only - HAVEN            | virus_host_prediction_test | [cov-s-host-prediction-multi-uniref90-sarscov2-variants.yaml](input/config-files/interpretability/sarscov2_variants/cov-s-host-prediction-multi-uniref90-sarscov2-variants.yaml)                       |
| Few Shot Learning  to predict unseen and rare hosts  | few_shot_learning          | [uniref90-fine-tuning-host-prediction-non-idv-multi-few-shot-learning.yaml](input/config-files/few_shot_learning/novel_host/uniref90-fine-tuning-host-prediction-non-idv-multi-few-shot-learning.yaml) |
| Few Shot Learning to predict hosts of unseen viruses | few_shot_learning          | [uniref90-fine-tuning-host-prediction-idv-multi-few-shot-learning.yaml](input/config-files/few_shot_learning/novel_virus/uniref90-fine-tuning-host-prediction-idv-multi-few-shot-learning.yaml)        |
| Generate embeddings of protein sequences using HAVEN | embedding_generation       | [uniref90-fine-tuning-host-prediction-multi-embedding.yaml](input/config-files/interpretability/embedding/uniref90-fine-tuning-host-prediction-multi-embedding.yaml)                                   |



---
## Data
All datasets used to pretrain and fine-tune HAVEN are available in [Zenodo](https://doi.org/10.5281/zenodo.15540220).

Information on the dataset preprocessing steps and script is available in the [Dataset Preprocessing Pipeline README](dataset_preprocessing_pipeline_README.md).

---
## Models
Download the pretrained and fine-tuned models of HAVEN from [Zenodo](https://doi.org/10.5281/zenodo.15537800).

---
## License
HAVEN software is available for public use under the GNU General Public License v3. 

---

[//]: # (## Reference)

[//]: # (If you use HAVEN in your research, please cite the following preprint:)

[//]: # ()
[//]: # (**<insert-bioRxiv-link-here>**)