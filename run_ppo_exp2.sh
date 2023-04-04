#!/bin/bash  

CASE="E2000v2"
DEV=1999
JOBS=10

python -u "generate_dataset.py" --name $CASE --typeDS VALIDATION --n_devices $DEV --n_jobs $JOBS --len_dataset 50 --np_seed_dataset 2023 &&
python -u "generate_dataset.py" --name $CASE --typeDS TEST --n_devices $DEV --n_jobs $JOBS --len_dataset 30 --np_seed_dataset 1290 &&
nohup python -u "train_ppo.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --max_updates 150 --device cuda > fp55_b.out 2> fp55.err < /dev/null &&
touch DONE_1.out &&



