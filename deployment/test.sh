#!/bin/bash

#SBATCH -N1 
#SBATCH --ntasks-per-node=12 # 1 node, 12 threads/cpus
#SBATCH -t 00:30 # 10 secs 
#SBATCH -p dev_q
#SBATCH -A personal

# Go to script deployment dir
HOME=/home/andrewclchan211/BioNLP
cd $HOME
pwd

# # Training:
# # python ~/BioNLP/bin/flu.py bilstm --train --test > flu_train.log 2>&1

