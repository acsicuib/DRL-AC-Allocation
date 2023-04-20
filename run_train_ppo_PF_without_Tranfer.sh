#!/bin/bash  

CASE="E1000v9"
DEV=999
JOBS=9

#rm *.out || true &&
#rm *.err || true &&
python -u "generate_dataset.py" --name $CASE --typeDS VALIDATION --n_devices $DEV --n_jobs $JOBS --len_dataset 50 --np_seed_dataset 2023 &&
python -u "generate_dataset.py" --name $CASE --typeDS TEST --n_devices $DEV --n_jobs $JOBS --len_dataset 30 --np_seed_dataset 1290 &&
nohup python -u "train_ppo.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --max_updates 150 --device cuda > out/fPF_5050_$CASE.out 2> out/fPF_55_$CASE.out < /dev/null &&
nohup python -u "train_ppo.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --max_updates 150 --device cpu --rewardWeightCost 1.0 --rewardWeightTime 0.0 --device cuda --record_alloc_episodes 0 10 20 30  > out/fPF_010_$CASE.out2> out/fPF_010_$CASE.err < /dev/null &&
nohup python -u "train_ppo.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --max_updates 100 --device cpu --rewardWeightCost 0.0 --rewardWeightTime 1.0 --device cuda --record_alloc_episodes 0 10 20 30 40 > out/fPF_010_$CASE.out 2> out/fPF_100_$CASE.err < /dev/null &&
nohup python -u "train_ppo.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --max_updates 150 --device cpu --rewardWeightCost 0.75 --rewardWeightTime 0.25 --device cuda > out/fPF_2575_$CASE.out 2> out/fPF_2575_$CASE.err < /dev/null &&
nohup python -u "train_ppo.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --max_updates 150 --device cpu --rewardWeightCost 0.25 --rewardWeightTime 0.75 --device cuda > out/fPF_7525_$CASE.err 2> out/fPF_7525_$CASE.err < /dev/null &&
touch DONE_1.out &&
nohup python -u "eval_trained_PF_ppo.py" --name $CASE  --n_devices $DEV --n_jobs $JOBS >  out/fPF_$CASE.out 2> out/fPF_$CASE.err < /dev/null &&
touch DONE_2.out &


#Pendiente >>
python -u "generate_dataset.py" --name E1000v9 --typeDS VALIDATION --n_devices 999 --n_jobs 9 --len_dataset 50 --np_seed_dataset 2023
python -u "generate_dataset.py" --name E1000v9 --typeDS TEST --n_devices 999 --n_jobs 9 --len_dataset 30 --np_seed_dataset 1290
nohup python -u "train_ppo.py" --name E1000v9 --n_devices 999 --n_jobs 9 --max_updates 150 --record_alloc_episodes 0 20 30 40  --device cuda > out/fPF_5050_E1000v9.out 2> out/fPF_55_E1000v9.err < /dev/null &
