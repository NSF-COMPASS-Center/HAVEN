# Executing VirProBERT: Language Model for Virus Host Prediction
This readme provides information on how to execute VirProBERT on Virginia Tech servers namely - 
- pandemic-da.cs.vt.edu
- arc.vt.edu

## Advanced Research Computing at Virginia Tech (ARC):
Please refer to the [ARC User Documentation](https://www.docs.arc.vt.edu/index.html) to learn about ARC and its usage.

### Get access
- Request [Dr. T. M. Murali](mailto:murali@cs.vt.edu) to add you to the `seqevol` projects for both _compute_ and _project_ allocations.
- Verify your access at https://coldfront.arc.vt.edu/. 
  - Home &rarr; "Allocation >>" 
  - There should be two allocations for "Models of sequence evolution" with "Compute" and "Project" resource names.

### Setup project
- SSH into one of the ARC GPU login nodes. As of February 25, 2025, the list of login nodes includes -
    ```
    tinkercliffs1.arc.vt.edu
    tinkercliffs2.arc.vt.edu
    infer1.arc.vt.edu
    falcon1.arc.vt.edu
    falcon2.arc.vt.edu
    ```
- Setup the [Project and Environment](#project-and-environment-setup)

- Create a folder named your Virginia Tech PID within `/projects/seqevol`. This folder will contain the input and output data files for all the experiments.
```shell
mkdir <vt-pid>
```
- Create input and output folders
```shell
cd /projects/seqevol/<vt-pid>
mkdir -p zoonosis/input/data
mkdir zoonosis/output
```
- Create symbolic links to these newly created input data and output folders from your zoonosis project.
```shell
cd <path-to-your-git-repo>/zoonosis
ln -s /projects/seqevol/<vt-pid>/zoonosis/input/data input/data
ln -s /projects/seqevol/<vt-pid>/zoonosis/output output
```

- Copy the necessary input data and output model files for your experiments from the file paths mentioned in [Data](#data)
- If you create any new input datasets for your experiments:
  - Upload the dataset files into `/projects/seqevol/<vt-pid>zoonosis/input/data`
  - Add entries for those files in the table in [Input data files](#input-data-files) so that they can be used by others. 
- All the newly created output files will automatically be persisted in `/projects/seqevol/<vt-pid>/zoonosis/output/raw`.
  - If you exceute an experiment that creates a model that will be used by others (for example, pre-trained model), add an entry in the table [Pre-trained and fine-tuned models](#pre-trained-and-fine-tuned-models)

### Executing experiments using batch jobs
Choose the deployment script based on the use-case

Examples
- [run_pipeline_gpu](deployment/arc/run_pipeline_gpu.sh)
```shell
sbatch deployment/arc/run_pipeline_gpu.sh . <path-to-the-config-file>
```
Example
```shell
sbatch deployment/arc/run_pipeline_gpu.sh . input/config-files/transfer_learning/fine_tuning/uniref90-fine-tuning-host-prediction-multi.yaml
```

- [run_script_gpu](deployment/arc/run_script_gpu.sh)
```shell
sbatch deployment/arc/run_script_gpu.sh . <path-to-script-to-executed> <arguments-required-by-the-script>
```

- Execute [perturbation_dataset_generator.py](src/utils/scripts/perturbation_dataset_generator.py) using [run_script_gpu](deployment/arc/run_script_gpu.sh)
```shell
sbatch deployment/arc/run_script_gpu.sh . src/utils/scripts/perturbation_dataset_generator.py -if input/data/coronaviridae/20240313/sarscov2/uniprot/coronaviridae_s_uniprot_sars_cov_2.csv -od input/data/coronaviridae/20240313/sarscov2/uniprot/perturbation_dataset/multi -st protein
```

## Pandemic-DA Server
- Email [Blessy Antony](mailto:blessyantony@vt.edu) to get access to the pandemic-da server.
- SSH into `pandemic-da.cs.vt.edu`. Reset your password after first login.
- Setup the [Project and Environment](#project-and-environment-setup)
- Input data files and pre-trained models are located at
```shell
<Insert path to project folder>
```
### Executing experiments using batch jobs
- Request [Dr. Anuj Karpatne](mailto:karpatne@vt.edu) to get you added to the pandemic-da server management Slack channel. 
All users of the server coordinate the usage of the four available GPUs using this Slack channel.
- Check the availability of GPUs by executing 
  ```
  nvidia-smi
  ```
- Execute virus host prediction pipeline using [run_pipeline_gpu.sh](deployment/pandemic-da/run_pipeline_gpu.sh)
  ```shell
  deployment/pandemic-da/run_pipeline_gpu.sh <path-to-config-file> <comma-separated-list-of-gpu-ids-to-be-used>
  ```
  Example
  ```shell
  screen
  conda activate virprobert
  deployment/pandemic-da/run_pipeline_gpu.sh input/config-files/transfer_learning/fine_tuning/uniref90-fine-tuning-host-prediction-multi.yaml 0
  
  OR
  
  deployment/pandemic-da/run_pipeline_gpu.sh input/config-files/transfer_learning/fine_tuning/uniref90-fine-tuning-host-prediction-multi.yaml 1,2
  ```
  
## Project and Environment Setup
### Code and dependencies
- Clone the GitHub repository at the desired location.
- Setup conda environment 
    ```shell
    bash
    conda create -n virprobert python=3.11.8
    conda activate virprobert
    pip install -r requirements.txt
    ```
- [Install pytorch](https://pytorch.org/get-started/locally/) based on the CUDA version in the server.

### Setup Weights & Biases
1. Create an account in [Weights & Biases](https://wandb.ai/site/).
2. Create a new project in Weights and Biases named `zoonosis-host-prediction`.
3. Set up the `wandb` library by completing [Step 1 in the Quickstart](https://wandb.ai/quickstart?utm_source=app-resource-center&utm_medium=app&utm_term=quickstart).
    - Note: Do not forget to log in to Weights and Biases (`wandb login`) in the server where you intend to execute the experiment.

## Data
Input protein sequence data used for all virus host prediction experiments are located at `/projetcs/seqevol`.

### Input data files

| <div style="width:600px">Dataset Description</div>                                                                                                                                    | Path                                                                                                                                                                                        |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Uniref90 viral protein sequences                                                                                                                                                      | `/projects/seqevol/blessyantony/zoonosis/input/data/uniref90/20240131/uniref90_viridae.csv`                                                                                                 |
| Uniref90 viral protein sequences with virus-hosts                                                                                                                                     | `/projects/seqevol/blessyantony/zoonosis/input/data/uniref90/20240131/uniref90_viridae_embl_hosts_pruned.csv`                                                                               |
| Uniref90 viral protein sequences with virus-hosts that belong to _vertebrata_ (are vertebrates)                                                                                       | `/projects/seqevol/blessyantony/zoonosis/input/data/uniref90/20240131/uniref90_viridae_embl_hosts_pruned_metadata_species_vertebrates.csv`                                                  |
| Uniref90 viral protein sequences with vertebrate virus hosts and from non-immunodeficiency viruses                                                                                    | `/projects/seqevol/blessyantony/zoonosis/input/data/uniref90/20240131/uniref90_viridae_embl_hosts_pruned_metadata_species_vertebrates_idv_hosts.csv`                                        |
| Uniref90 viral protein sequences with vertebrate virus hosts and from non-immunodeficiency viruses with prevalence $\ge$ 1% (common hosts)                                            | `/projects/seqevol/blessyantony/zoonosis/input/data/uniref90/20240131/uniref90_viridae_embl_hosts_pruned_metadata_species_vertebrates_w_seq_non_idv_t0.01_c5.csv`                           |
| Uniref90 viral protein sequences with vertebrate virus hosts and from non-immunodeficiency viruses with prevalence $\ge$ 1% (common hosts) and sequence length within 99.9 percentile | `/projects/seqevol/blessyantony/zoonosis/input/data/uniref90/20240131/uniref90_viridae_embl_hosts_pruned_metadata_species_vertebrates_w_seq_non_idv_t0.01_c5_seq_len_in_99.9percentile.csv` |
| Uniref90 viral protein sequences with vertebrate virus hosts and from non-immunodeficiency viruses with prevalence $\ge$ 0.05% and $<$ 1% (rare hosts)                                | `/projects/seqevol/blessyantony/zoonosis/input/data/uniref90/20240131/uniref90_viridae_embl_hosts_pruned_metadata_species_vertebrates_w_seq_non_idv_lt_1_gte_0.05_prcnt_prevalence.csv`     |
| Uniref90 viral protein sequences with vertebrate virus hosts and from immunodeficiency viruses (unseen virus)                                                                         | `/projects/seqevol/blessyantony/zoonosis/input/data/uniref90/20240131/uniref90_viridae_embl_hosts_pruned_metadata_species_vertebrates_idv_hosts.csv`                                        | 

### Pre-trained and fine-tuned models
| <div style="width:600px">Model Description</div>                                                                                                                                                          | File Path                                                                                                                                                                                                                                                                                               |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| VirProBERT: Pre-trained model(segment length=256)                                                                                                                                                         | `/projects/seqevol/blessyantony/zoonosis/output/raw/uniref90-viridae/pre-training/mlm/20240821/transformer_encoder-l_6-h_8-lr1e-4_msl256_b512_splitseq_mlm_vs30cls_allemb_itr0.pth`                                                                                                                     |
| VirProBERT: Fine-tuned on Uniref90 non-immunodeficiency virus common classes dataset for virus-host prediction (5 classes)                                                                                | `/projects/seqevol/blessyantony/zoonosis/output/raw/uniref90_embl_vertebrates_non_idv_t0.01_c5_seq_len_in_99.9percentile/20240826/host_multi/fine_tuning_hybrid_cls/mlm_tfenc_l6_h8_lr1e-4_uniref90viridae_msl256s64allemb_vs30cls_batchnorm_hybrid_attention_msl256s64ae_fnn_2l_d1024_lr1e-4_itr4.pth` |
| VirProBERT: Fine-tuned on Uniref90 non-immunodeficiency virus common classes dataset for virus-host prediction (5 classes) and fine-tuned using Few-shot learning (3-way, 5-shot) to predict rare classes | `/projects/seqevol/blessyantony/zoonosis/output/raw/uniref90_embl_vertebrates_non_idv/20240928/host_multi/few_shot_learning/fsl_tr_w3s5q10_te_w3s5q-1_e100b32_split70-10-20_hybrid-attention_sl256st64vs30cls_fnn_2l_d1024_lr1e-4_itr4.pth`                                                             |