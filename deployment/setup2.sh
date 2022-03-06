#!/bin/bash

# ----------------------------------------
# Desc: 
# Meant to be run on Tinkerclifs login node

# Usage:
# /bin/bash -i ./setup2.sh
# Prepares environment for tasks

# Prereqs:
# Project must be in the home dir.
# Run /bin/bash -i ./setup1.sh before this

# ----------------------------------------

CONDA_ENV_NAME="BioNLP"

# Go to script dir
cd "$(dirname "$0")"

module purge
module load apps site/tinkercliffs-rome/easybuild/setup Anaconda3

# Conda environment 3.9
cd ..
conda create -y -n $CONDA_ENV_NAME python=3.9
conda activate $CONDA_ENV_NAME

pip install -r requirements.txt

# Install data
wget http://cb.csail.mit.edu/cb/viral-mutation/data.tar.gz
tar xvf data.tar.gz

