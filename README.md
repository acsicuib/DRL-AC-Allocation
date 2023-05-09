# Project: Multi-objective optimization problem of application placement in the cloud continuum

This project is focused to solve a multi-objective optimization problem of application placement in the cloud continuum using a deep reinforcement learning (DRL) approach. Unlike other optimization techniques such as ILP or evolutionary algorithms, DRL models can be applied in real-time to solve similar problem situations after being trained. 

Our multi-objective model is composed of a learning process that involves a graph neural network and two actor-critics to have a holistic view of the priorities in terms of interconnected services that compose an application and a scalar problem decomposition process to compute the dominant solutions for each objective. 

The learning model considers the relationships between services as a key factor when selecting their placement: services with higher dependencies should have priority when choosing their locations. The scalar decomposition involves training multiple models according to the balance of the objectives pursued in the problem.

The project contains complementary experimentation with baseline strategies and with a single-objective GA and a multi-objective such as NSGA-II.

This repository contains the code presented on this published paper:

```bibtex
@inproceedings{
title={},
authors="",
booktitle={""},
publisher={""},
year={2023}, 
pages={-}
}
```
## Structure 

The study use three algorithms: our PPO approach, a NSGAII and a GA algorithm

Folders:
- environment/ The DRL environment definition
- models/ The ppo definition
- datasets/ Contains the test and train datasets
- logs/ Contains the logs files to be analysed from shell executions of approaches commands 
- out/ Contains the stdout and stderr from shell executions
- savedModels/ Contains the ppo models recorded
- GAmodel/ has NSGA and GA versions of the fog placement problem

Main files:
- eval_*.py the main algorithms for find solutions in the dataset
- train_*.py the learning algorithm 
- run_*.sh shell files to run previous algorithms in different parameters configurations
- notebook_*.ipynb files to analyse the results
- parameters.py contains all the paramters used as default arguments in the algorithms
- instance_generator.py generates one instance of our application based on JSSP 
- generate_dataset.py generates a whole dataset of instances

### Installation

We recomend the use of a python virtual environment:

```bash
pip install -r requirements.txt
```

### How to run it?

You can run sh files or you can execute specific scripts:

Take care in the number of updates, generations or the specification of a CUDA device to train the different approaches.

```bash
nohup python -u "train_ppo.py" --name "TEST" --n_devices 999 --n_jobs 9 --max_updates 150 --num_layers 5 --num_mlp_layers_actor 5 --num_mlp_layers_critic 3 --k_epochs 1 --device cuda > out/fp55_TEST.out 2> out/fp55_TEST.err < /dev/null
```




