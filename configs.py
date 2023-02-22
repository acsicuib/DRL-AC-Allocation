
import numpy as np


et_normalize_coef = 1000
wkr_normalize_coef = 100
rewardscale = 0
device = "cpu"


# Scenarios
num_envs = 1
n_jobs = 3
n_tasks = n_jobs**2
n_machines = 4 #+1 the Cloud entity

#Experiments #epocs
max_updates = 1

# Instance generator
task_time_low = 3
task_time_high = 20

feat_labels = ["Processing time","Cost","Lat","Load"]#,"Load Penalty"] 
#procesing time -> big Better
#lat -> small better
n_feat = len(feat_labels)
r_n_feat = 2+1 #processing time, latency, allocation ## TODO. fijar número final de features usadas en el State

cpu_speed = [2,4,6]
cost_options = [1,3]
latency_options = [1,5,10] #invertir category max-> min
# cloud_features = np.array([20,1,30,10,0]) 
cloud_features = np.array([10.,1.,30.,10.]) 


# Torch 
torch_seed = 2022
np_seed_train = 2023

# Model
# lr_agent  = 2e-5 #agent
# lr_critic = 1e-4 #critic

lr_agent  = 2e-2 #agent
lr_critic = 1e-2 #critic


gamma=1 # discount factor
k_epochs=1 # update policy for K epochs
eps_clip=0.2 # clip parameter for PPO
vloss_coef = 1 # critic loss coefficient
ploss_coef = 2 # policy loss coefficient
entloss_coef = 0.01 # 'entropy loss coefficient


# args for network
num_layers=3 # No. of layers of feature extraction GNN including input layer
neighbor_pooling_type='sum' # neighbour pooling type
input_dim=r_n_feat # number of dimension of raw node features
hidden_dim=64 # hidden dim of MLP in fea extract GNN


num_mlp_layers_feature_extract=3 #No. of layers of MLP in fea extract GNN
num_mlp_layers_feature_hw = 3
num_mlp_layers_actor=3 # No. of layers in actor MLP
hidden_dim_actor=32 # 'hidden dim of MLP in actor

num_mlp_layers_critic=3 # No. of layers in critic MLP
hidden_dim_critic=32 # hidden dim of MLP in critic

learn_eps = False
graph_pool_type='average' # graph pooling type
decay_step_size=2000 # decay_step_size
decay_ratio = 0.9
decayflag = False #lr decayflag

# args placement network
input_dim_PL = (r_n_feat-1)+n_tasks*2