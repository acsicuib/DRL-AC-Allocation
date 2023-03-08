#!/bin/sh
echo "Test 1"
python parameters.py --name="test1"
echo "Test 2"
python parameters.py --name="test2" --num_envs 10 --n_jobs 3 --n_devices 99 --latency_options 1 5 10 20 
