#!/bin/bash

#SBATCH -J haven
#SBATCH --account=seqevol
#SBATCH --partition=a100_normal_q

#SBATCH --mem=180G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -t 6:00:00 # wall-time required (# 96hrs)


# Load modules
module reset
module load
# Load conda
module load Miniconda3
#Load CUDA
module load CUDA/12.6.0

# Load conda environment
source activate haven
echo "Conda information:"
conda info

# Setup project and result directories
PROJECT_DIR=$1
LOGS_DIR=$PROJECT_DIR/output/logs
export PYTHONPATH="$PROJECT_DIR/src"
echo "Project directory: $PROJECT_DIR"

# Execute python script
SCRIPT_LOCATION=$2
shift # shift all arguments one to the left. So $1 is dropped, $1 is now original $2 and so on and so forth
shift # shift all arguments one to the left again. So $2 is dropped this time, $1 is now original $3 and so on and so forth
ARGS="$@" # all the remaining args

LOG_FILE=$LOGS_DIR/$(date +%Y_%b_%d_%H_%M_%s).log
echo "Log File: $LOG_FILE"
echo "Script START"
date
python $SCRIPT_LOCATION $ARGS > $LOG_FILE 2>&1
echo "Script END"
date


