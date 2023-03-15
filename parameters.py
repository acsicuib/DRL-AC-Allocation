import argparse

feature_labels = ["Processing time","Cost","Latency","Load"]


parser = argparse.ArgumentParser(description='Arguments')
### Case study name
parser.add_argument('--name', type=str, default="default", help='Case study name to record results')
parser.add_argument('--record_ppo', type=bool, default=True, help='Record log of PPO evolution loss values per episode')
parser.add_argument('--record_alloc', type=bool, default=True, help='Record log of placements per episode')
parser.add_argument('--record_alloc_episodes', type=int, nargs='+', default=[0,10,20,40], help='Episodes where it gets a log of current placements')

### Environment parameters
parser.add_argument('--rewardscale', type=int, default=0, help='Reward increment/decrement in actions with same behaviour in St > St+1')
parser.add_argument('--et_normalize_coef', type=int, default=1000, help='Coeficient factor to normalize application execution time')

### Experimental parameters
parser.add_argument('--num_envs', type=int, default=40, help='Number of environments/trajectories in each episode')
parser.add_argument('--n_jobs', type=int, default=3, help='Number of jobs')
parser.add_argument('--n_devices', type=int, default=9, help='Number of fog devices WITHOUT the Cloud-entity. TotalDevices = n_machines+1')
parser.add_argument('--max_updates', type=int, default=1, help='Number of episodes')

### Application args
parser.add_argument('--DAG_rand_dependencies_factor', type=float, default=0.3, help='Probability factor of a task of having a dependency with some predecesor tasks')
parser.add_argument('--task_time_low',  type=int, default=3,  help='Minumun range value of the operation units of a task')
parser.add_argument('--task_time_high', type=int, default=20, help='Maximun range value of the operation units of a task')

### Device args
parser.add_argument('--latency_options', nargs='+', type=int, default=[1,5,10,15,20,25,30], help="A sequence of intergers. Latency values for device generation. Smaller better.")
parser.add_argument('--cpu_speed_options', nargs='+', type=int, default=[2,4,6], help="A sequence of intergers. CPU speed values for device generation. Bigger better.")
parser.add_argument('--cost_options', nargs='+', type=int, default=[1,3], help="A sequence of intergers. Cost values for device generation. Smaller better.")
### Feature Cloud-entity
parser.add_argument('--cloud_features', nargs='+', type=int, default=[10,1,30,0], help="Cloud entity features: %s"%feature_labels)

### Torch 
parser.add_argument('--torch_seed', type=int, default=2022, help='Torch seed')
parser.add_argument('--np_seed_train', type=int, default=2023, help='Numpy seed')
parser.add_argument('--device', type=str, default="cpu", help='Device')


### GNN model
# parser.add_argument('--num_layers', type=int, default= 3, help='Number of MLPs for featuring extraction at GNN')
parser.add_argument('--num_layers', type=int, default= 4, help='Number of MLPs for featuring extraction at GNN')
parser.add_argument('--neighbor_pooling_type', type=str, default="sum", help='Neighbour pooling type')
parser.add_argument('--hidden_dim', type=int, default= 64, help='Hidden dims in the MLP (GNN model)')
parser.add_argument('--num_mlp_layers_feature_extract', type=int, default= 3, help='Number of layers for each MLP (GNN model)')
parser.add_argument('--graph_pool_type', type=str, default='average', help='Average weight of neighbourd extraction feature at GNN')

### Agent-Critic models. Both ACs work with the same parameters 
# parser.add_argument('--lr_agent', type=float, default= 2e-2, help='Actor learning factor')
parser.add_argument('--lr_agent', type=float, default=0.022, help='Actor learning factor')
parser.add_argument('--lr_critic', type=float, default= 0.0269, help='Critic learning factor')
# parser.add_argument('--lr_critic', type=float, default= 1e-2, help='Critic learning factor')
# parser.add_argument('--num_mlp_layers_actor', type=int, default= 3, help='MLP layers in the Actor network')
parser.add_argument('--num_mlp_layers_actor', type=int, default= 5, help='MLP layers in the Actor network')
parser.add_argument('--num_mlp_layers_critic', type=int, default= 3, help='MLP layers in the Critic network')
parser.add_argument('--hidden_dim_actor', type=int, default= 32, help='Hidden dimension of the Actor network')
parser.add_argument('--hidden_dim_critic', type=int, default= 32, help='Hidden dimension of the Critic network')

### RL model
parser.add_argument('--gamma', type=float, default= 1, help='RL discount factor')
parser.add_argument('--k_epochs', type=int, default= 2, help='Number of episodes to update the policy')
# parser.add_argument('--k_epochs', type=int, default= 1, help='Number of episodes to update the policy')
parser.add_argument('--eps_clip', type=float, default= 0.25, help='Actor-critics clip factor')
# parser.add_argument('--eps_clip', type=float, default= 0.2, help='Actor-critics clip factor')
# parser.add_argument('--vloss_coef', type=float, default= 1, help='Critics loss coefficient')
parser.add_argument('--vloss_coef', type=float, default= 3, help='Critics loss coefficient')
parser.add_argument('--ploss_coef', type=float, default= 2, help='Actor-critics loss coefficient')
# parser.add_argument('--entloss_coef', type=float, default= 0.01, help='Entropy loss coefficient')
parser.add_argument('--entloss_coef', type=float, default= 0.023, help='Entropy loss coefficient')

### RL - Scheduler
parser.add_argument('--decay_step_size', type=int, default= 2000, help='Decay step ratio')
parser.add_argument('--decay_ratio', type=float, default= 0.9, help='Decay ratio')
parser.add_argument('--decayflag', type=bool, default=False, help='Apply decay step')


# Transforms arguments to a global variable
configs = parser.parse_args()

# TODO IMPROVE the generation and placement of these vars
configs.n_tasks = configs.n_jobs**2
configs.feature_labels = feature_labels

# TODO FIX at the end of this research project
configs.r_n_feat = 2+1 #processing time, latency, +allocation  # Real number of features used in the model 
configs.input_dim=configs.r_n_feat # GNN imput dimension (x)-> GIN 
configs.input_dim_device = (configs.r_n_feat-1)+configs.n_tasks*2 #Second Actor-Critic input dimension (x)-> AC_2


if __name__ == '__main__':
    from parameters import configs
    print(configs)
    # python arg.py --nargs-int-type 1234 2345 3456 4567

