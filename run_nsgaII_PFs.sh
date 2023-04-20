#!/bin/bash
CASE="E1000v150"
DEV=999
JOBS=9
python -u "generate_dataset.py" --name $CASE --typeDS VALIDATION --n_devices $DEV --n_jobs $JOBS --len_dataset 50 --np_seed_dataset 2023 &&
python -u "generate_dataset.py" --name $CASE --typeDS TEST --n_devices $DEV --n_jobs $JOBS --len_dataset 30 --np_seed_dataset 1290 &&
nohup python -u "eval_nsgaII.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --n_gen 150 > out/fcPF_$CASE.out 2> out/fPF.err < /dev/null &&
CASE="E1000v200"
python -u "generate_dataset.py" --name $CASE --typeDS VALIDATION --n_devices $DEV --n_jobs $JOBS --len_dataset 50 --np_seed_dataset 2023 &&
python -u "generate_dataset.py" --name $CASE --typeDS TEST --n_devices $DEV --n_jobs $JOBS --len_dataset 30 --np_seed_dataset 1290 &&
nohup python -u "eval_nsgaII.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --n_gen 200 > out/fcPF_$CASE.out 2> out/fPF.err < /dev/null &&
CASE="E1000v250"
python -u "generate_dataset.py" --name $CASE --typeDS VALIDATION --n_devices $DEV --n_jobs $JOBS --len_dataset 50 --np_seed_dataset 2023 &&
python -u "generate_dataset.py" --name $CASE --typeDS TEST --n_devices $DEV --n_jobs $JOBS --len_dataset 30 --np_seed_dataset 1290 &&
nohup python -u "eval_nsgaII.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --n_gen 250 > out/fcPF_$CASE.out 2> out/fPF.err < /dev/null &&

#!/bin/bash
CASE="E1000v300"
DEV=999
JOBS=9
python -u "generate_dataset.py" --name $CASE --typeDS VALIDATION --n_devices $DEV --n_jobs $JOBS --len_dataset 50 --np_seed_dataset 2023 &&
python -u "generate_dataset.py" --name $CASE --typeDS TEST --n_devices $DEV --n_jobs $JOBS --len_dataset 30 --np_seed_dataset 1290 &&
nohup python -u "eval_nsgaII.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --n_gen 300 > out/fcPF_$CASE.out 2> out/fPF.err < /dev/null &&
touch out/DONE.out