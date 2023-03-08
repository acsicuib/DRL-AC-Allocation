#!/bin/sh
echo "Case study 0"
python ppo_train.py --name "cs0" --num_envs 10 --n_jobs 3 --n_devices 9 --max_updates 1
echo "Plotting?"

echo "Done"

