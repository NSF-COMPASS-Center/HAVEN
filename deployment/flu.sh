#!/bin/bash

#SBATCH -J hello-world

#SBATCH -N 1 
#SBATCH --ntasks-per-node=12 # 1 node, 12 threads/cpus
#SBATCH -t 4:00:00 # 4 hours

#SBATCH -p a100_normal_q
# #SBATCH -A arcadm

#SBATCH --gres=gpu:1

# Go to script deployment dir
cd "$(dirname "$0")"
cd ..

module purge
module load apps site/tinkercliffs-rome/easybuild/setup Anaconda3

source activate ~/.conda/envs/BioNLP

# Run python 
python bin/flu.py bilstm --checkpoint models/flu.hdf5 --embed > flu_embed.log 2>&1

python bin/flu.py bilstm --checkpoint models/flu.hdf5 --semantics > flu_semantics.log 2>&1

python bin/flu.py bilstm --checkpoint models/flu.hdf5 --combfit > flu_combfit.log 2>&1


# # Training:
# # python ~/BioNLP/bin/flu.py bilstm --train --test > flu_train.log 2>&1




