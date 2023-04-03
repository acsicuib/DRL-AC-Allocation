#!/bin/bash

CASE="E500"
DEV=499
JOBS=6

nohup python -u "eval_nsgaII_mono.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --n_gen 100 --rewardWeightCost 0.5 --rewardWeightTime 0.5 > fM.out 2> feM.err < /dev/null &&
nohup python -u "eval_nsgaII_mono.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --n_gen 100 --rewardWeightCost 1. --rewardWeightTime 0. > fM2.out 2> feM2.err < /dev/null &&
nohup python -u "eval_nsgaII_mono.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --n_gen 100 --rewardWeightCost 0. --rewardWeightTime 1. > fL2.out 2> feL2.err < /dev/null &&
nohup python -u "eval_nsgaII_mono.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --n_gen 100 --rewardWeightCost 0.25 --rewardWeightTime 0.75 > fN2.out 2> feN2.err < /dev/null &&
nohup python -u "eval_nsgaII_mono.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --n_gen 100 --rewardWeightCost 0.75 --rewardWeightTime 0.25 > fK2.out 2> feK2.err < /dev/null &