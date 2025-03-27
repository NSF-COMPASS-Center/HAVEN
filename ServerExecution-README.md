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
- Input data files and pre-trained models are located at
```shell
cd /projects/seqevol
```
- Create a folder named your Virginia Tech PID within `/projects/seqevol`
```shell
mkdir <vt-pid>
```
- Create input and output folders
```shell
cd /projects/seqevol/<vt-pid>
mkdir -p zoonosis/input/data
mkdir zoonosis/output
```
Upload any newly created dataset files into `/projects/seqevol/<vt-pid>zoonosis/input/data`

- Create symbolic links to these newly created input data and output folders from your zoonosis project.
```shell
cd <path-to-your-git-repo>/zoonosis
ln -s /projects/seqevol/<vt-pid>/zoonosis/input/data input/data
ln -s /projects/seqevol/<vt-pid>/zoonosis/output output
```

All the newly created output files will automatically be persisted in `/projects/seqevol/<vt-pid>zoonosis/output/`

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
    conda create -n virprobert python=3.10
    conda activate virprobert
    pip install -r requirements.txt
    ```
- Copy the necessary input dataset files and pre-trained models from the appropriate locations from each server.

### Create required folders
Create a directory for logs in the output directory using - 
```shell
mkdir -p output/logs
```

### Setup Weights & Biases
1. Create an account in [Weights & Biases](https://wandb.ai/site/).
2. Create a new project in Weights and Biases named `zoonosis-host-prediction`.
3. Setup the `wandb` library by completing [Step 1 in the Quickstart](https://wandb.ai/quickstart?utm_source=app-resource-center&utm_medium=app&utm_term=quickstart).
    - Note: Do not forget to log in to Weights and Biases (`wandb login`) in the server where you intend to execute the experiment.