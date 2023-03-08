
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Arguments')


parser.add_argument('--device', type=str, default="cpu", help='Device')

### Environment parameters

parser.add_argument('--rewardscale', type=int, default=0, help='Reward increment/decrement in actions with same behaviour in St > St+1')
parser.add_argument('--et_normalize_coef', type=int, default=1000, help='Coeficient factor to normalize application execution time')

### Experimental parameters
parser.add_argument('--num_envs', type=int, default=40, help='Number of environments/trajectories in each episode')
parser.add_argument('--n_jobs', type=int, default=9, help='Number of jobs')
parser.add_argument('--n_devices', type=int, default=9, help='Number of fog devices WITHOUT the Cloud-entity. TotalDevices = n_machines+1')
parser.add_argument('--max_updates', type=int, default=100, help='Number of episodes')

#### Application args
parser.add_argument('--DAG_rand_dependencies_factor', type=float, default=0.3, help='Probability factor of a task of having a dependency with some predecesor tasks')
parser.add_argument('--task_time_low',  type=int, default=3,  help='Minumun range value of the operation units of a task')
parser.add_argument('--task_time_high', type=int, default=20, help='Maximun range value of the operation units of a task')

### Device args
parser.add_argument('--latency_options', nargs='+', type=int, default=[1,5,10,15,20,30], help="A sequence of intergers. Latency values for device generation.")
parser.add_argument('--cpu_speed_options', nargs='+', type=int, default=[2,4,6], help="A sequence of intergers. CPU speed values for device generation.")
parser.add_argument('--cost_options', nargs='+', type=int, default=[1,3], help="A sequence of intergers. Cost values for device generation.")


### Torch 
parser.add_argument('--torch_seed', type=int, default=2022, help='Torch seed')
parser.add_argument('--np_seed_train', type=int, default=2023, help='Numpy seed')


##N GNN model
parser.add_argument('--num_layers', type=int, default= 3, help='Number of MLPs for featuring extraction at GNN')
parser.add_argument('--neighbor_pooling_type', type=str, default="sum", help='Neighbour pooling type')
parser.add_argument('--hidden_dim', type=int, default= 64, help='Hidden dims in the MLP (GNN model)')

parser.add_argument('--num_mlp_layers_feature_extract', type=int, default= 3, help='Number of layers for each MLP (GNN model)')

parser.add_argument('--hidden_dim', type=int, default= 64, help='Hidden dims in the MLP (GNN model)')

### Agent-Critic models. Both ACs work with the same parameters 
parser.add_argument('--lr_agent', type=float, default= 2e-2, help='Actor learning factor')
parser.add_argument('--lr_critic', type=float, default= 1e-2, help='Critic learning factor')
parser.add_argument('--num_mlp_layers_actor', type=int, default= 3, help='MLP layers in the Actor network')
parser.add_argument('--num_mlp_layers_critic', type=int, default= 3, help='MLP layers in the Critic network')
parser.add_argument('--hidden_dim_actor', type=int, default= 32, help='Hidden dimension of the Actor network')
parser.add_argument('--hidden_dim_critic', type=int, default= 32, help='Hidden dimension of the Critic network')

### RL model

parser.add_argument('--gamma', type=float, default= 1, help='RL discount factor')
parser.add_argument('--k_epochs', type=int, default= 1, help='Number of episodes to update the policy')
parser.add_argument('--eps_clip', type=float, default= 0.2, help='Actor-critics clip factor')
parser.add_argument('--vloss_coef', type=float, default= 1, help='Critics loss coefficient')
parser.add_argument('--ploss_coef', type=float, default= 2, help='Actor-critics loss coefficient')
parser.add_argument('--entloss_coef', type=float, default= 0.01, help='Entropy loss coefficient')



## Transforms arguments to a global variable
configs = parser.parse_args()

### Final customization #TODO IMPROVE
n_tasks = n_jobs**2
feat_labels = ["Processing time","Cost","Lat","Load"]#,"Load Penalty"] 
#procesing time -> big Better
#lat -> small better
n_feat = len(feat_labels)
r_n_feat = 2+1 #processing time, latency, allocation
cloud_features = np.array([10.,1.,70.,10.])  

input_dim=r_n_feat # number of dimension of raw node features

# cpu_speed = [2,4,6]

# latency_options = [1,5,10,15,20,25,30,35,40,45,50,55,60,65,70] #invertir category max-> min
# latency_options = [1,5,10,15,20,30] #invertir category max-> min
# cloud_features = np.array([20,1,30,10,0]) 
# cloud_features = np.array([10.,1.,70.,10.])  #TODO dinamyc Cloud generation


# Model
# lr_agent  = 2e-5 #agent
# lr_critic = 1e-4 #critic


# args for network

learn_eps = False
graph_pool_type='average' # graph pooling type
decay_step_size=2000 # decay_step_size
decay_ratio = 0.9
decayflag = False #lr decayflag

# args placement network
input_dim_PL = (r_n_feat-1)+n_tasks*2


if __name__ == '__main__':
    import configs
    # python arg.py --nargs-int-type 1234 2345 3456 4567