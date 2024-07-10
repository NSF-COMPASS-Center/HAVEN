#!/bin/bash

echo "Conda information:"
conda info


# Execute python script
SCRIPT_LOCATION=./src/zoonosis.py
CONFIG_FILE=$1
GPU_DEVICE=$2

LOGS_DIR=output/logs
LOG_FILE=$LOGS_DIR/$(date +%Y_%b_%d_%H_%M_%s).log

echo "Config File: $CONFIG_FILE"
echo "Log File: $LOG_FILE"

# set GPU device
export CUDA_VISIBLE_DEVICES=$GPU_DEVICE
echo "GPU check"
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}\n Available GPU devices: {torch.cuda.device_count()}')"
echo "Zoonosis NLP models START"
date
python $SCRIPT_LOCATION -c $CONFIG_FILE > $LOG_FILE 2>&1
echo "Zoonosis NLP models END"
date


