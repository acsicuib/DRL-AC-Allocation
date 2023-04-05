#!/bin/bash

CASE="E500v6"
DEV=499
JOBS=6

python -u "generate_dataset.py" --name $CASE --typeDS VALIDATION --n_devices $DEV --n_jobs $JOBS --len_dataset 50 --np_seed_dataset 2023 &&
python -u "generate_dataset.py" --name $CASE --typeDS TEST --n_devices $DEV --n_jobs $JOBS --len_dataset 30 --np_seed_dataset 1290 &&
nohup python -u "eval_nsgaII_mono.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --n_gen 100 --rewardWeightCost 0.5 --rewardWeightTime 0.5 > out/fa_$CASE.out 2> out/fa.err < /dev/null &&
touch out/DONE__$CASE.out &&
nohup python -u "eval_nsgaII.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --n_gen 100 > out/fPF_$CASE.out 2> out/fPF.err < /dev/null &&
touch out/DONE2__$CASE.out &&

CASE="E500v9"
JOBS=9

python -u "generate_dataset.py" --name $CASE --typeDS VALIDATION --n_devices $DEV --n_jobs $JOBS --len_dataset 50 --np_seed_dataset 2023 &&
python -u "generate_dataset.py" --name $CASE --typeDS TEST --n_devices $DEV --n_jobs $JOBS --len_dataset 30 --np_seed_dataset 1290 &&
nohup python -u "eval_nsgaII_mono.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --n_gen 100 --rewardWeightCost 0.5 --rewardWeightTime 0.5 > out/fa_$CASE.out 2> out/fa.err < /dev/null &&
touch out/DONE__$CASE.out &&
nohup python -u "eval_nsgaII.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --n_gen 100 > out/fPF_$CASE.out 2> out/fPF.err < /dev/null &&
touch out/DONE2__$CASE.out &&

CASE="E500v12"
JOBS=12

python -u "generate_dataset.py" --name $CASE --typeDS VALIDATION --n_devices $DEV --n_jobs $JOBS --len_dataset 50 --np_seed_dataset 2023 &&
python -u "generate_dataset.py" --name $CASE --typeDS TEST --n_devices $DEV --n_jobs $JOBS --len_dataset 30 --np_seed_dataset 1290 &&
nohup python -u "eval_nsgaII_mono.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --n_gen 100 --rewardWeightCost 0.5 --rewardWeightTime 0.5 > out/fa_$CASE.out 2> out/fa.err < /dev/null &&
touch out/DONE__$CASE.out &&
nohup python -u "eval_nsgaII.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --n_gen 100 > out/fPF_$CASE.out 2> out/fPF.err < /dev/null &&
touch out/DONE2__$CASE.out &&


CASE="E1000v9"
DEV=999
JOBS=9

python -u "generate_dataset.py" --name $CASE --typeDS VALIDATION --n_devices $DEV --n_jobs $JOBS --len_dataset 50 --np_seed_dataset 2023 &&
python -u "generate_dataset.py" --name $CASE --typeDS TEST --n_devices $DEV --n_jobs $JOBS --len_dataset 30 --np_seed_dataset 1290 &&
nohup python -u "eval_nsgaII_mono.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --n_gen 100 --rewardWeightCost 0.5 --rewardWeightTime 0.5 > out/fa_$CASE.out 2> out/fa.err < /dev/null &&
touch out/DONE__$CASE.out &&
nohup python -u "eval_nsgaII.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --n_gen 100 > out/fPF_$CASE.out 2> out/fPF.err < /dev/null &&
touch out/DONE2__$CASE.out &&

CASE="E1000v12"
DEV=999
JOBS=12

python -u "generate_dataset.py" --name $CASE --typeDS VALIDATION --n_devices $DEV --n_jobs $JOBS --len_dataset 50 --np_seed_dataset 2023 &&
python -u "generate_dataset.py" --name $CASE --typeDS TEST --n_devices $DEV --n_jobs $JOBS --len_dataset 30 --np_seed_dataset 1290 &&
nohup python -u "eval_nsgaII_mono.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --n_gen 100 --rewardWeightCost 0.5 --rewardWeightTime 0.5 > out/fa_$CASE.out 2> out/fa.err < /dev/null &&
touch out/DONE__$CASE.out &&
nohup python -u "eval_nsgaII.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --n_gen 100 > out/fPF_$CASE.out 2> out/fPF.err < /dev/null &&
touch out/DONE2__$CASE.out &&


CASE="E1500v9"
DEV=1499
JOBS=9

python -u "generate_dataset.py" --name $CASE --typeDS VALIDATION --n_devices $DEV --n_jobs $JOBS --len_dataset 50 --np_seed_dataset 2023 &&
python -u "generate_dataset.py" --name $CASE --typeDS TEST --n_devices $DEV --n_jobs $JOBS --len_dataset 30 --np_seed_dataset 1290 &&
nohup python -u "eval_nsgaII_mono.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --n_gen 100 --rewardWeightCost 0.5 --rewardWeightTime 0.5 > out/fa_$CASE.out 2> out/fa.err < /dev/null &&
touch out/DONE__$CASE.out &&
nohup python -u "eval_nsgaII.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --n_gen 100 > out/fPF_$CASE.out 2> out/fPF.err < /dev/null &&
touch out/DONE2__$CASE.out &&

CASE="E1500v12"
JOBS=12

python -u "generate_dataset.py" --name $CASE --typeDS VALIDATION --n_devices $DEV --n_jobs $JOBS --len_dataset 50 --np_seed_dataset 2023 &&
python -u "generate_dataset.py" --name $CASE --typeDS TEST --n_devices $DEV --n_jobs $JOBS --len_dataset 30 --np_seed_dataset 1290 &&
nohup python -u "eval_nsgaII_mono.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --n_gen 100 --rewardWeightCost 0.5 --rewardWeightTime 0.5 > out/fa_$CASE.out 2> out/fa.err < /dev/null &&
touch out/DONE__$CASE.out &&
nohup python -u "eval_nsgaII.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --n_gen 100 > out/fPF_$CASE.out 2> out/fPF.err < /dev/null &&
touch out/DONE2__$CASE.out &&


CASE="E2000v9"
DEV=1999
JOBS=9

python -u "generate_dataset.py" --name $CASE --typeDS VALIDATION --n_devices $DEV --n_jobs $JOBS --len_dataset 50 --np_seed_dataset 2023 &&
python -u "generate_dataset.py" --name $CASE --typeDS TEST --n_devices $DEV --n_jobs $JOBS --len_dataset 30 --np_seed_dataset 1290 &&
nohup python -u "eval_nsgaII_mono.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --n_gen 100 --rewardWeightCost 0.5 --rewardWeightTime 0.5 > out/fa_$CASE.out 2> out/fa.err < /dev/null &&
touch out/DONE__$CASE.out &&
nohup python -u "eval_nsgaII.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --n_gen 100 > out/fPF_$CASE.out 2> out/fPF.err < /dev/null &&
touch out/DONE2__$CASE.out &&


CASE="E2000v12"
DEV=1999
JOBS=12

python -u "generate_dataset.py" --name $CASE --typeDS VALIDATION --n_devices $DEV --n_jobs $JOBS --len_dataset 50 --np_seed_dataset 2023 &&
python -u "generate_dataset.py" --name $CASE --typeDS TEST --n_devices $DEV --n_jobs $JOBS --len_dataset 30 --np_seed_dataset 1290 &&
nohup python -u "eval_nsgaII_mono.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --n_gen 100 --rewardWeightCost 0.5 --rewardWeightTime 0.5 > out/fa_$CASE.out 2> out/fa.err < /dev/null &&
touch out/DONE__$CASE.out &&
nohup python -u "eval_nsgaII.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --n_gen 100 > out/fPF_$CASE.out 2> out/fPF.err < /dev/null &&
touch out/DONE2__$CASE.out &&

CASE="E2000v15"
DEV=1999
JOBS=15

python -u "generate_dataset.py" --name $CASE --typeDS VALIDATION --n_devices $DEV --n_jobs $JOBS --len_dataset 50 --np_seed_dataset 2023 &&
python -u "generate_dataset.py" --name $CASE --typeDS TEST --n_devices $DEV --n_jobs $JOBS --len_dataset 30 --np_seed_dataset 1290 &&
nohup python -u "eval_nsgaII_mono.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --n_gen 100 --rewardWeightCost 0.5 --rewardWeightTime 0.5 > out/fa_$CASE.out 2> out/fa.err < /dev/null &&
touch out/DONE__$CASE.out &&
nohup python -u "eval_nsgaII.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --n_gen 100 > out/fPF_$CASE.out 2> out/fPF.err < /dev/null &&
touch out/DONE2__$CASE.out &