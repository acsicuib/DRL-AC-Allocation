nohup python -u "eval_nsgaII_mono.py" --name E100 --n_devices 99 --n_jobs 3 --n_gen 100 --rewardWeightCost 0.5 --rewardWeightTime 0.5 > f55.out 2> fe55.err < /dev/null &&
nohup python -u "eval_nsgaII_mono.py" --name E100 --n_devices 99 --n_jobs 3 --n_gen 100 --rewardWeightCost 1. --rewardWeightTime 0. > f10.out 2> fe10.err < /dev/null &&
nohup python -u "eval_nsgaII_mono.py" --name E100 --n_devices 99 --n_jobs 3 --n_gen 100 --rewardWeightCost 0. --rewardWeightTime 1. > f01.out 2> fe01.err < /dev/null &&
nohup python -u "eval_nsgaII_mono.py" --name E100 --n_devices 99 --n_jobs 3 --n_gen 100 --rewardWeightCost 0.25 --rewardWeightTime 0.75 > f27.out 2> fe27.err < /dev/null &&
nohup python -u "eval_nsgaII_mono.py" --name E100 --n_devices 99 --n_jobs 3 --n_gen 100 --rewardWeightCost 0.75 --rewardWeightTime 0.25 > f72.out 2> fe72.err < /dev/null &
