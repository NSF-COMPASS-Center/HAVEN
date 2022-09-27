#!/bin/bash

#SBATCH -J BILSTM_HOST_HEP
#SBATCH -A seqevol
#SBATCH -N1 
#SBATCH -t 12:00:00 # n hours

#SBATCH -p a100_normal_q
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=128G # Yeah, ik making all those permutations costs a lot of memory, which is the required format for TF, unless I want to alter the TF libs, this is fine

#SBATCH --export=NONE # Fixes some bugs with pathing

# Relative path vars, since SLURM doesn't seem to preserve original env's env vars
USER_HOME=$HOME
PROJECT_DIR=$PWD
echo "Current working directory: $PROJECT_DIR"

# module reset
# module load Anaconda3
# module load cuDNN/8.1.1.33-CUDA-11.2.1

# source activate $USER_HOME/.conda/envs/BioNLP

python --version

# Parameters
SCRIPT_LOCATION=$PROJECT_DIR/src/pipeline.py
MODEL=bilstm
RESULTS_DIR=$PROJECT_DIR/results
CONFIG_DIR=$PROJECT_DIR/config-files

echo "Job start"
date
echo "Log File: $RESULTS_DIR/hep_host_notransfer.$(date +%Y_%b_%d_%H_%M).log"

# Ensure results directory exists
mkdir -p $RESULTS_DIR

python $SCRIPT_LOCATION -c $CONFIG_DIR/hepGenotypeNoTransfer.yaml > $RESULTS_DIR/hep_genotype_notransfer.$(date +%Y_%b_%d_%H_%M).log 2>&1

echo "Job done"
date
