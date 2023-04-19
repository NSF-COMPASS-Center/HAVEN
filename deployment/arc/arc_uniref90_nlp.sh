#!/bin/bash

#SBATCH -J zoonosis-uniref90-nlp
#SBATCH --account=seqevol
#SBATCH --partition=a100_normal_q

#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -t 72:00:00 # wall-time required (# 72hrs)


# Load modules
module reset
module load
# Load conda
module load Anaconda3
#Load CUDA
module load cuda11.2/toolkit

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
LOG_FILE=$LOGS_DIR/$3.$(date +%Y_%b_%d_%H_%M).log
echo "Config File: $CONFIG_FILE"
echo "Log File: $LOG_FILE"

echo "GPU check"
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"
echo "Zoonosis NLP models START"
date
~/anaconda3/envs/zoonosis/bin/python $SCRIPT_LOCATION -c $CONFIG_FILE > $LOG_FILE 2>&1
echo "Zoonosis NLP models END"
date


