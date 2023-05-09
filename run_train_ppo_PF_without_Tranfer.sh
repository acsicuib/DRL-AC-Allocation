#!/bin/bash  

CASE="E1000v9"
DEV=999
JOBS=9

python -u "generate_dataset.py" --name $CASE --typeDS VALIDATION --n_devices $DEV --n_jobs $JOBS --len_dataset 50 --np_seed_dataset 2023 &&
python -u "generate_dataset.py" --name $CASE --typeDS TEST --n_devices $DEV --n_jobs $JOBS --len_dataset 30 --np_seed_dataset 1290 &&
nohup python -u "train_ppo.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --max_updates 150 --device cuda > out2/fPF_5050_$CASE.out 2> out2/fPF_55_$CASE.out < /dev/null &&
nohup python -u "train_ppo.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --max_updates 150 --device cuda --rewardWeightCost 1.0 --rewardWeightTime 0.0  --record_alloc_episodes 0 10 20 30  > out2/fPF_010_$CASE.out2> out2/fPF_010_$CASE.err < /dev/null &&
nohup python -u "train_ppo.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --max_updates 150 --device cuda --rewardWeightCost 0.0 --rewardWeightTime 1.0  --record_alloc_episodes 0 10 20 30 40 > out2/fPF_010_$CASE.out 2> out2/fPF_100_$CASE.err < /dev/null &&
nohup python -u "train_ppo.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --max_updates 150 --device cuda --rewardWeightCost 0.75 --rewardWeightTime 0.25  > out2/fPF_2575_$CASE.out 2> out2/fPF_2575_$CASE.err < /dev/null &&
nohup python -u "train_ppo.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --max_updates 150 --device cuda --rewardWeightCost 0.25 --rewardWeightTime 0.75  > out2/fPF_7525_$CASE.err 2> out2/fPF_7525_$CASE.err < /dev/null &&
touch DONE_1.out &&
nohup python -u "eval_trained_PF_ppo.py" --name $CASE  --n_devices $DEV --n_jobs $JOBS >  out2/fPF_$CASE.out 2> out2/fPF_$CASE.err < /dev/null &&
touch DONE_2.out &

#Pendiente >>
python -u "generate_dataset.py" --name E1000v9 --typeDS VALIDATION --n_devices 999 --n_jobs 9 --len_dataset 50 --np_seed_dataset 2023
python -u "generate_dataset.py" --name E1000v9 --typeDS TEST --n_devices 999 --n_jobs 9 --len_dataset 30 --np_seed_dataset 1290
nohup python -u "train_ppo.py" --name E1000v9 --n_devices 999 --n_jobs 9 --max_updates 150 --record_alloc_episodes 0 20 30 40  --device cuda > out2/fPF_5050_E1000v9.out 2> out2/fPF_55_E1000v9.err < /dev/null &
