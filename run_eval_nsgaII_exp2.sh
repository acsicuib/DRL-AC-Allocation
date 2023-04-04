#!/bin/bash

CASE="E1500v2"
DEV=1499
JOBS=9
CASE2="E2000v2"
DEV2=1999
JOBS2=10


python -u "generate_dataset.py" --name $CASE --typeDS VALIDATION --n_devices $DEV --n_jobs $JOBS --len_dataset 50 --np_seed_dataset 2023 &&
python -u "generate_dataset.py" --name $CASE --typeDS TEST --n_devices $DEV --n_jobs $JOBS --len_dataset 30 --np_seed_dataset 1290 &&
python -u "generate_dataset.py" --name $CASE2 --typeDS VALIDATION --n_devices $DEV2 --n_jobs $JOBS2 --len_dataset 50 --np_seed_dataset 2023 &&
python -u "generate_dataset.py" --name $CASE2 --typeDS TEST --n_devices $DEV2 --n_jobs $JOBS2 --len_dataset 30 --np_seed_dataset 1290 &&
nohup python -u "eval_nsgaII_mono.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --n_gen 100 --rewardWeightCost 0.5 --rewardWeightTime 0.5 > fM_1500.out 2> feM.err < /dev/null &&
nohup python -u "eval_nsgaII_mono.py" --name $CASE2 --n_devices $DEV2 --n_jobs $JOBS2 --n_gen 100 --rewardWeightCost 0.5 --rewardWeightTime 0.5 > fM_2000.out 2> feM.err < /dev/null &&
touch DONE_PF.out &&
nohup python -u "eval_nsgaII.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --n_gen 100 > fPF.out 2> fPF.err < /dev/null &&
nohup python -u "eval_nsgaII.py" --name $CASE2 --n_devices $DEV2 --n_jobs $JOBS2 --n_gen 100 > fPF.out 2> fPF.err < /dev/null &&
touch DONE.out &