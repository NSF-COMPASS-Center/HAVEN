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

set -e

CONDA_ENV_NAME="BioNLP"

# Go to script dir
cd "$(dirname "$0")"

echo "Something?"
module purge
module load apps site/tinkercliffs-rome/easybuild/setup Anaconda3
echo "Something 1?"

# Conda environment 3.9
cd ..
echo "Something 2?"
conda create -y -n $CONDA_ENV_NAME python=3.9
echo "Conda create worked?"
conda activate $CONDA_ENV_NAME
echo "Conda activate worked?"

pip install -r requirements.txt
echo "pip install worked?"

# Install data
wget http://cb.csail.mit.edu/cb/viral-mutation/data.tar.gz
tar xvf data.tar.gz
echo "data install worked?"

