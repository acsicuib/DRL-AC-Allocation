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
        nohup python -u "train_ppo.py" --name $CASE --n_devices $DEV --n_jobs $JOBS --max_updates 150 --device cuda > out/fp55_$CASE.out 2> out/fp55_$CASE.err < /dev/null &&
        touch out/DONE_ppo_$CASE.out
	done
done