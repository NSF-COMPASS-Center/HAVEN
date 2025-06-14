#!/bin/bash

#SBATCH -J haven
#SBATCH --account=seqevol
#SBATCH --partition=normal_q

#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH -t 48:00:00 # wall-time required (# 24hrs)


# Load modules
module reset
module load
module load Miniconda3


# Load conda environment
source activate haven
echo "Conda information:"
conda info

# Setup project and result directories
PROJECT_DIR=$1
LOGS_DIR=$PROJECT_DIR/output/logs
echo "Project directory: $PROJECT_DIR"
echo "Python version in $CONDA_PREFIX"

conda activate haven
echo "Python version in $CONDA_PREFIX"
$CONDA_PREFIX/bin/python --version

# Execute python script
SCRIPT_LOCATION=$PROJECT_DIR/src/run.py
CONFIG_FILE=$2
LOG_FILE=$LOGS_DIR/$(date +%Y_%b_%d_%H_%M_%s).log
echo "Config File: $CONFIG_FILE"
echo "Log File: $LOG_FILE"

echo "Pipeline START"
date
$CONDA_PREFIX/bin/python $SCRIPT_LOCATION -c $CONFIG_FILE > $LOG_FILE 2>&1
echo "Pipeline END"
date


