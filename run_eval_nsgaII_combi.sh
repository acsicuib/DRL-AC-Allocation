#!/bin/bash

CASE="E1000v2Combi"
DEV=999
JOBS=9

python -u "generate_dataset.py" --name $CASE --typeDS VALIDATION --n_devices $DEV --n_jobs $JOBS --len_dataset 50 --np_seed_dataset 2023 &&
python -u "generate_dataset.py" --name $CASE --typeDS TEST --n_devices $DEV --n_jobs $JOBS --len_dataset 30 --np_seed_dataset 1290 &&
nohup python -u "eval_nsgaII_mono.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --n_gen 100 --rewardWeightCost 0.5 --rewardWeightTime 0.5 > out/fc1_$CASE.out 2> out/fa.err < /dev/null &&
nohup python -u "eval_nsgaII_mono.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --n_gen 100 --rewardWeightCost 1. --rewardWeightTime 0. > out/fc2_$CASE.out 2> out/fa.err < /dev/null &&
nohup python -u "eval_nsgaII_mono.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --n_gen 100 --rewardWeightCost 0. --rewardWeightTime 1. > out/fc3_$CASE.out 2> out/fa.err < /dev/null &&
nohup python -u "eval_nsgaII_mono.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --n_gen 100 --rewardWeightCost 0.25 --rewardWeightTime 0.75 > out/fc5_$CASE.out 2> out/fa.err < /dev/null &&
nohup python -u "eval_nsgaII_mono.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --n_gen 100 --rewardWeightCost 0.75 --rewardWeightTime 0.25 > out/fc6_$CASE.out 2> out/fa.err < /dev/null &&
# touch out/DONE_PF.out &&
# nohup python -u "eval_nsgaII.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --n_gen 100 > out/fcPF_$CASE.out 2> out/fPF.err < /dev/null &&
touch out/DONE.out &