#! /bin/bash

#SBATCH -J BILSTM_FLU_MODEL
#SBATCH -A seqevol
#SBATCH -N1 
#SBATCH --ntasks-per-node=128 # 1 node, use all the power of 128 cores/threads/cpus RYZEN EPYC 7702 on tinkerclifs
#SBATCH -t 03:30:00 # 3 hour, 30 mins

#SBATCH -p a100_dev_q
#SBATCH --gres=gpu:1

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
SCRIPT_LOCATION=$PROJECT_DIR/bin/hiv.py 
MODEL=bilstm
SAVED_MODEL=$PROJECT_DIR/models/hiv.hdf5 
RESULTS_DIR=$PROJECT_DIR/results

# Ensure results directory exists
mkdir -p $RESULTS_DIR

# Run python scripts
python $SCRIPT_LOCATION $MODEL --checkpoint $SAVED_MODEL --embed > $RESULTS_DIR/hiv_embed.log 2>&1
python $SCRIPT_LOCATION $MODEL --checkpoint $SAVED_MODEL --semantics > $RESULTS_DIR/hiv_semantics.log 2>&1
python $SCRIPT_LOCATION $MODEL --checkpoint $SAVED_MODEL --combfit > $RESULTS_DIR/hiv_combfit.log 2>&1


# # Training:
# # python ~/BioNLP/bin/flu.py bilstm --train --test > flu_train.log 2>&1




