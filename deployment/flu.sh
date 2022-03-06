#!/bin/bash

#SBATCH -J hello-world

#SBATCH -N 1 
#SBATCH --ntasks-per-node=12 # 1 node, 12 threads/cpus
#SBATCH -t 0:20 # 1/3 minutes

#SBATCH -p a100_normal_q
# #SBATCH -A arcadm

#SBATCH --gres=gpu:1

source activate ~/.conda/envs/BioNLP

# Run python 
python ~/NLP/bin/flu.py bilstm --checkpoint models/flu.hdf5 --embed > ~/NLP/flu_embed.log 2>&1


