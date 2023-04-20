#!/bin/bash
for JOBS in 9 12
do
	for DEV in 1499 1999
	do
		CASE=$((DEV+1))
        CASE="${CASE}v${JOBS}"
        echo $CASE
        #echo $JOBS
        #echo $DEV
       	python -u "generate_dataset.py" --name $CASE --typeDS VALIDATION --n_devices $DEV --n_jobs $JOBS --len_dataset 50 --np_seed_dataset 2023
        python -u "generate_dataset.py" --name $CASE --typeDS TEST --n_devices $DEV --n_jobs $JOBS --len_dataset 30 --np_seed_dataset 1290
        nohup python -u "eval_nsgaII_mono.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --n_gen 100 --rewardWeightCost 0.5 --rewardWeightTime 0.5 > out/fa_$CASE.out 2> out/fa_$CASE.err < /dev/null
        touch out/DONE__$CASE.out
        nohup python -u "eval_nsgaII.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --n_gen 100 > out/fPF_$CASE.out 2> out/fPF_$CASE.err < /dev/null &&
        touch out/DONE2__$CASE.out
	done
done