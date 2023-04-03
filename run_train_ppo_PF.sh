#!/bin/bash  

CASE="E1000v2"
DEV=999
JOBS=9


rm *.out &&
rm *.err &&
python -u "generate_dataset.py" --name $CASE --typeDS VALIDATION --n_devices $DEV --n_jobs $JOBS --len_dataset 50 --np_seed_dataset 2023 &&
python -u "generate_dataset.py" --name $CASE --typeDS TEST --n_devices $DEV --n_jobs $JOBS --len_dataset 30 --np_seed_dataset 1290 &&
nohup python -u "train_ppo.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --max_updates 150 --device cuda > fp55.out 2> fp55.err < /dev/null &&
nohup python -u "train_ppo.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --max_updates 150 --device cpu --rewardWeightCost 1.0 --rewardWeightTime 0.0 --device cuda --record_alloc_episodes 0 10 20 30  > fp01.out 2> fp01.err < /dev/null &&
nohup python -u "train_ppo.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --max_updates 100 --device cpu --rewardWeightCost 0.0 --rewardWeightTime 1.0 --device cuda --record_alloc_episodes 0 10 20 30 40 > fp10.out 2> fp10.err < /dev/null &&
nohup python -u "train_ppo.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --max_updates 150 --device cpu --rewardWeightCost 0.75 --rewardWeightTime 0.25 --device cuda > fp27.out 2> fp27.err < /dev/null &&
nohup python -u "train_ppo.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --max_updates 150 --device cpu --rewardWeightCost 0.25 --rewardWeightTime 0.75 --device cuda > fp72.out 2> fp72.err < /dev/null &&
touch DONE_1.out &&
nohup python -u "eval_trained_PF_ppo.py" --name $CASE  --n_devices $DEV --n_jobs $JOBS > fpEND.out 2> fpEND.err < /dev/null &&
touch DONE_2.out &
