#!/bin/bash
JOBS=9
DEV=999
CASE=$((DEV+1))
GEN=100
CASE="${CASE}v${JOBS}v${GEN}"
echo $CASE
python -u "generate_dataset.py" --name $CASE --typeDS VALIDATION --n_devices $DEV --n_jobs $JOBS --len_dataset 50 --np_seed_dataset 2023
python -u "generate_dataset.py" --name $CASE --typeDS TEST --n_devices $DEV --n_jobs $JOBS --len_dataset 30 --np_seed_dataset 1290
nohup python -u "eval_nsgaII_mono.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --n_gen $GEN --rewardWeightCost 0.5 --rewardWeightTime 0.5 > out/fa_$CASE.out 2> out/fa_$CASE.err < /dev/null &&
nohup python -u "eval_nsgaII_mono.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --n_gen $GEN --rewardWeightCost 0.75 --rewardWeightTime 0.25 > out/fb_$CASE.out 2> out/fa_$CASE.err < /dev/null &&
nohup python -u "eval_nsgaII_mono.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --n_gen $GEN --rewardWeightCost 0.25 --rewardWeightTime 0.75 > out/fc_$CASE.out 2> out/fa_$CASE.err < /dev/null &&
nohup python -u "eval_nsgaII_mono.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --n_gen $GEN --rewardWeightCost 1.0 --rewardWeightTime 0.0 > out/fd_$CASE.out 2> out/fa_$CASE.err < /dev/null &&
nohup python -u "eval_nsgaII_mono.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --n_gen $GEN --rewardWeightCost 0.0 --rewardWeightTime 1.0 > out/fd_$CASE.out 2> out/fa_$CASE.err < /dev/null &&