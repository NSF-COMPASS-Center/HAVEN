#!/bin/bash

#SBATCH -J virprobert
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


