#!/bin/bash
# Run this with bash -i ./setup1.sh before running next setup file
# You should exit and re-ssh after using this script.

# Load libs
module purge
module load apps site/tinkercliffs-rome/easybuild/setup Anaconda3

# Init conda
conda init

# Kick user off ssh
exit
