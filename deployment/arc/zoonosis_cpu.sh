#!/bin/bash

#SBATCH -J zoonosis-uniref90-baseline
#SBATCH --account=seqevol
#SBATCH --partition=normal_q

#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH -t 48:00:00 # wall-time required (# 24hrs)


# Load modules
module reset
module load
module load Anaconda3


# Load conda environment
source activate ~/anaconda3/envs/zoonosis
echo "Conda information:"
conda info

# Setup project and result directories
PROJECT_DIR=$1
LOGS_DIR=$PROJECT_DIR/output/logs
echo "Project directory: $PROJECT_DIR"

# Execute python script
SCRIPT_LOCATION=$PROJECT_DIR/src/zoonosis.py
CONFIG_FILE=$2
LOG_FILE=$LOGS_DIR/$(date +%Y_%b_%d_%H_%M_%s).log
echo "Config File: $CONFIG_FILE"
echo "Log File: $LOG_FILE"

echo "Zoonosis baseline model START"
date
~/anaconda3/envs/zoonosis/bin/python $SCRIPT_LOCATION -c $CONFIG_FILE > $LOG_FILE 2>&1
echo "Zoonosis baseline model END"
date


