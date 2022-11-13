#!/bin/bash

#SBATCH -J zoonosis-hev-baseline
#SBATCH --account=seqevol
#SBATCH --partition=normal_q

#SBATCH --mem=32G
#SBATCH -N1 --ntasks-per-node=4 # number of nodes and number of tasks per node
#SBATCH -t 08:00:00 # wall-time required (
# 8hrs)
#SBATCH --mem-per-cpu=32G


# Load modules
module reset
module load
module load Anaconda3


# Load conda environment
source activate ~/anaconda3/envs/zoonosis-baseline
echo "Conda information:"
conda info

# Setup project and result directories
PROJECT_DIR=$1
LOGS_DIR=$PROJECT_DIR/output/logs
mkdir -p $RESULTS_DIR #ensure that the results directory exists
echo "Project directory: $PROJECT_DIR"

# Execute python script
SCRIPT_LOCATION=$PROJECT_DIR/src/protein_structure_analysis.py
CONFIG_FILE=$2
LOG_FILE=$LOGS_DIR/$3.$(date +%Y_%b_%d_%H_%M).log
echo "Config File: $CONFIG_FILE"osi
echo "Log File: $LOG_FILE"

echo "Zoonosis baseline model START"
date
python $SCRIPT_LOCATION -c $CONFIG_FILE > $LOG_FILE 2>&1
echo "Zoonosis baseline model END"
date


