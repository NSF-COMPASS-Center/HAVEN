#!/bin/bash

#SBATCH -J zoonosis-hep-bilstm-nolstm-host
#SBATCH --account=seqevol
#SBATCH --partition=a100_normal_q

#SBATCH --gres gpu:1
#SBATCH -N1 -- ntasks-per-node=1 # number of nodes
#SBATCH -t 00:02:00 # time required


# Load modules
module reset
module load
module load Anaconda3
module load cuDNN/8.1.1.33-CUDA-11.2.1
source activate zoonosis-bilstm


PROJECT_DIR=$1
echo "Current working directory: $PROJECT_DIR"

python --version

# Results directory
RESULTS_DIR=$PROJECT_DIR/results
mkdir -p $RESULTS_DIR #ensure that the results directory exists

# Parameters
SCRIPT_LOCATION=$PROJECT_DIR/src/pipeline.py
CONFIG_FILE=$2
echo "Config File: $CONFIG_FILE"
echo "Job start"
date
echo "Log File: $RESULTS_DIR/hep_host_notransfer.$(date +%Y_%b_%d_%H_%M).log"

echo "python $SCRIPT_LOCATION -c $CONFIG_FILE > $RESULTS_DIR/hep_host_notransfer.$(date +%Y_%b_%d_%H_%M).log 2>&1"

echo "Job done"
date


