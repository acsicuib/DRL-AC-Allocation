# Project: Multi-objective application placement in fog computing using graph neural network-based reinforcement learning

This project addresses the multi-objective optimization problem of application placement in the cloud continuum by leveraging a deep reinforcement learning (DRL) approach. Unlike traditional optimization techniques like integer linear programming (ILP) or evolutionary algorithms, DRL models can operate in real-time, providing solutions to similar problem scenarios once they are trained.

Our multi-objective model employs a learning process that integrates a graph neural network with two actor-critic components. This setup offers a comprehensive view of the priorities associated with the interconnected services that constitute an application. Additionally, a scalar problem decomposition method is used to compute dominant solutions for each objective.

The learning model takes into account the dependencies between services as a critical factor in placement decisions, prioritizing services with stronger interdependencies for optimal location selection. The scalar decomposition approach involves training multiple models that balance the objectives of the problem, depending on their relative importance.

The project also includes complementary experimentation, comparing the DRL approach with baseline strategies, a single-objective genetic algorithm (GA), and a multi-objective algorithm such as NSGA-II.

This repository contains the code presented in this published paper:

```bibtex
@article{lera_multi-objective_2024,
	title = {Multi-objective application placement in fog computing using graph neural network-based reinforcement learning},
	issn = {1573-0484},
	url = {https://doi.org/10.1007/s11227-024-06439-5},
	doi = {10.1007/s11227-024-06439-5},
	journal = {The Journal of Supercomputing},
	author = {Lera, Isaac and Guerrero, Carlos},
	month = aug,
	year = {2024},
}
```
## Structure 

The study uses three algorithms: our PPO approach, an NSGAII and a GA algorithm

Folders:
- environment/ The DRL environment definition
- models/ The ppo definition
- datasets/ Contains the test and train datasets
- logs/ Contains the logs files to be analysed from shell executions of approach commands 
- out/ Contains the stdout and stderr from shell executions
- savedModels/ Contains the ppo models recorded
- GAmodel/ has NSGA and GA versions of the fog placement problem

Main files:
- eval_*.py the main algorithms for finding solutions in the dataset
- train_*.py the learning algorithm 
- run_*.sh shell files to run previous algorithms in different parameter configurations
- notebook_*.ipynb files to analyse the results
- parameters.py contains all the parameters used as default arguments in the algorithms
- instance_generator.py generates one instance of our application based on JSSP 
- generate_dataset.py generates a whole dataset of instances

### Installation

We recommend the use of a Python virtual environment:

```bash
pip install -r requirements.txt
```

### How to run it?

You can run shell scripts or execute specific scripts directly.

Pay attention to the number of updates, the number of generations, and the specification of a CUDA device required for training different approaches.

```bash
nohup python -u "train_ppo.py" --name "TEST" --n_devices 999 --n_jobs 9 --max_updates 150 --num_layers 5 --num_mlp_layers_actor 5 --num_mlp_layers_critic 3 --k_epochs 1 --device cuda > out/fp55_TEST.out 2> out/fp55_TEST.err < /dev/null
```




