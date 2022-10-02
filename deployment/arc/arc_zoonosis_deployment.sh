#!/bin/bash

#SBATCH -J zoonosis-hep-bilstm-nolstm-host
#SBATCH --account=seqevol
#SBATCH --partition=a100_normal_q

#SBATCH --gres gpu:1
#SBATCH -N1 --ntasks-per-node=4 # number of nodes and number of tasks per node
#SBATCH -t 24:00:00 # time required
#SBATCH --mem-per-cpu=128G


# Load modules
module reset
module load
module load Anaconda3
module load cuda11.2/toolkit
module load TensorFlow


# Load conda environment
source activate ~/anaconda3/envs/zoonosis-bilstm
echo "Conda information:"
conda info

# Setup project and result directories
PROJECT_DIR=$1
RESULTS_DIR=$PROJECT_DIR/results
mkdir -p $RESULTS_DIR #ensure that the results directory exists
echo "Project directory: $PROJECT_DIR"
echo "Results directory: $RESULTS_DIR"

# Execute python script
SCRIPT_LOCATION=$PROJECT_DIR/src/pipeline.py
CONFIG_FILE=$2
LOG_FILE=$RESULTS_DIR/hep_host_notransfer.$(date +%Y_%b_%d_%H_%M).log
echo "Config File: $CONFIG_FILE"
echo "Log File: $LOG_FILE"

echo "Zoonosis bilstm model START"
date
python $SCRIPT_LOCATION -c $CONFIG_FILE > $LOG_FILE 2>&1
echo "Zoonosis bilstm model END"
date


