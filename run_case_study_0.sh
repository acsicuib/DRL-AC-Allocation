#!/bin/sh
echo "Case study 0"
python train_ppo.py --name "cs0" --num_envs 10 --n_jobs 3 --n_devices 9 --max_updates 1
echo "Plotting?"

echo "Done"

