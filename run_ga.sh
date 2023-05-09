CASE="E1000v9"
DEV=999
JOBS=9
#echo $DEV
python -u "generate_dataset.py" --name $CASE --typeDS VALIDATION --n_devices $DEV --n_jobs $JOBS --len_dataset 50 --np_seed_dataset 2023
python -u "generate_dataset.py" --name $CASE --typeDS TEST --n_devices $DEV --n_jobs $JOBS --len_dataset 30 --np_seed_dataset 1290
nohup python -u "eval_ga.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --n_gen 100 --rewardWeightCost 0.5 --rewardWeightTime 0.5 > out2/f_ga_w55_$CASE.out 2> out2/f_ga_a$CASE.err < /dev/null
nohup python -u "eval_ga.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --n_gen 100 --rewardWeightCost 0.75 --rewardWeightTime 0.25 > out2/f_ga_w72_$CASE.out 2> out2/f_ga_b$CASE.err < /dev/null
nohup python -u "eval_ga.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --n_gen 100 --rewardWeightCost 0.25 --rewardWeightTime 0.75 > out2/f_ga_w27_$CASE.out 2> out2/f_ga_c$CASE.err < /dev/null
nohup python -u "eval_ga.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --n_gen 100 --rewardWeightCost 0 --rewardWeightTime 1. > out2/f_ga_w01_$CASE.out 2> out2/f_ga_d$CASE.err < /dev/null
nohup python -u "eval_ga.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --n_gen 100 --rewardWeightCost 1. --rewardWeightTime 0. > out2/f_ga_w10_$CASE.out 2> out2/f_ga_e$CASE.err < /dev/null
