#!/bin/bash
# Run this with bash -i ./setup1.sh before running next setup file
module purge
module load apps site/tinkercliffs-rome/easybuild/setup Anaconda3

conda init
exit
