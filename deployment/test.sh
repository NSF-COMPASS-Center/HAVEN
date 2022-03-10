#!/bin/bash

#SBATCH -J hello-world

#SBATCH -N 1 
#SBATCH --ntasks-per-node=12 # 1 node, 12 threads/cpus
#SBATCH -t 00:10 # 10 secs 

#SBATCH -p a100_normal_q
#SBATCH -A arcadm

# Go to script deployment dir
cd "$(dirname "$0")"
cd ..
pwd

# # Training:
# # python ~/BioNLP/bin/flu.py bilstm --train --test > flu_train.log 2>&1




