#!/bin/sh
echo "Case study 0"
python train_ppo.py --name "Test0" --num_envs 10 --n_jobs 3 --n_devices 9 --max_updates 40
echo "Plotting?"

echo "Done"

