#!/usr/bin/env bash

cd ..
python bin/hep.py bilstm --train --test  > results/hep_train.log 2>&1
cd -
