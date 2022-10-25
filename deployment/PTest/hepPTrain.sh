#!/bin/bash

#SBATCH -J BILSTM_HOST_HEP
#SBATCH -A seqevol
#SBATCH -N1 
#SBATCH -t 24:00:00 # n hours

#SBATCH -p a100_normal_q
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=64G # Yeah, ik making all those permutations costs a lot of memory, which is the required format for TF, unless I want to alter the TF libs, this is fine

#SBATCH --export=NONE # Fixes some bugs with pathing

# Relative path vars, since SLURM doesn't seem to preserve original env's env vars
USER_HOME=/home/andrewclchan211
PROJECT_DIR=$USER_HOME/BioNLP
cd $PROJECT_DIR

module reset
module load Anaconda3
module load cuDNN/8.1.1.33-CUDA-11.2.1

source activate $USER_HOME/.conda/envs/BioNLP

python --version

# Parameters
SCRIPT_LOCATION=$PROJECT_DIR/src/pipeline.py
MODEL=bilstm
#SAVED_MODEL=$PROJECT_DIR/models/cov.hdf5 
SAVED_MODEL=$PROJECT_DIR/target/hep/checkpoints/bilstm/r1/bilstm_512-11.hdf5
RESULTS_DIR=$PROJECT_DIR/results
CONFIG_DIR=$PROJECT_DIR/config-files

echo "Job start"
date

# Ensure results directory exists
mkdir -p $RESULTS_DIR


#declare -a arr=("132197556" "187253406" "556209273" "692853522" "766510479")
declare -a arr=("132197556")

## now loop through the above array
for i in "${arr[@]}"; do
    for j in $CONFIG_DIR/PTest/$i/*.yaml; do
    echo "start $j $(date)"
    python $SCRIPT_LOCATION -c $j > $RESULTS_DIR/$(basename $j)-$(date +%Y_%b_%d_%H_%M).log 2>&1
    echo "Location: $RESULTS_DIR/$(basename $j)-$(date +%Y_%b_%d_%H_%M).log" 
    echo "finish $(date)"
    done
done

echo "Job done"
date
