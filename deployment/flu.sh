#! /bin/bash

#SBATCH -J BILSTM_FLU_MODEL
#SBATCH -A seqevol
#SBATCH -N1 
#SBATCH --ntasks-per-node=128 # 1 node, use all the power of 128 cores/threads/cpus RYZEN EPYC 7702 on tinkerclifs
#SBATCH -t 01:30:00 # 1 hour, 30 mins
#SBATCH -p a100_dev_q
#SBATCH --gres=gpu:1

# Relative path vars, since SLURM doesn't seem to preserve original env's env vars
HOME=/home/andrewclchan211
PROJECT_DIR=$HOME/BioNLP
cd $HOME

module purge
module load apps site/tinkercliffs-rome/easybuild/setup Anaconda3

source activate $HOME/.conda/envs/BioNLP

# Parameters
SCRIPT_LOCATION=$PROJECT_DIR/bin/flu.py 
MODEL=bilstm
SAVED_MODEL=$PROJECT_DIR/models/flu.hdf5 
RESULTS_DIR=$PROJECT_DIR/results

# Ensure results directory exists
mkdir -p $RESULTS_DIR

# Run python scripts
python $SCRIPT_LOCATION $MODEL --checkpoint $SAVED_MODEL --embed > $RESULTS_DIR/flu_embed.log 2>&1

#python $SCRIPT_LOCATION $MODEL --checkpoint $SAVED_MODEL --semantics > $RESULTS_DIR/flu_semantics.log 2>&1

#python $SCRIPT_LOCATION $MODEL --checkpoint $SAVED_MODEL --combfit > $RESULTS_DIR/flu_combfit.log 2>&1


# # Training:
# # python ~/BioNLP/bin/flu.py bilstm --train --test > flu_train.log 2>&1




