import torch
import numpy as np

from parameters import configs
from environment.env import *
from models.dag_aggregate import dag_pool


def validate_model(dataset, model):
    device = torch.device(configs.device)
    
    number_all_device_features = len(configs.feature_labels) #TODO fix 
    env = SPP(number_jobs=configs.n_jobs, number_devices=configs.n_devices,number_features=number_all_device_features) 

    dag_pool_step = dag_pool(graph_pool_type=configs.graph_pool_type,
                             batch_size=torch.Size([1, configs.n_tasks, configs.n_tasks]),
                             n_nodes=configs.n_tasks, device=device)
    
    log_rewards = []
    # rollout using model
    for sample in dataset:
        times, adj, feat = sample

        _, state, candidate, mask = env.reset(*sample)
        state_ft = state[0]
        state_fm = state[1]

        ep_reward = - env.getRewardInit()

        while True:
            adj_tensor_env = torch.from_numpy(adj).to(device).to_sparse()
            state_ft_tensor_env = torch.from_numpy(state_ft).to(device) 
            state_fm_tensor_env = torch.from_numpy(state_fm).to(device)
            candidate_tensor_env = torch.from_numpy(candidate).to(device)
            mask_tensor_env = torch.from_numpy(mask).to(device)
            
            with torch.no_grad():
                task_action, _, _, _, _, ix_machine_action, _, _, _  = model(state_ft=state_ft_tensor_env,
                              state_fm=state_fm_tensor_env.unsqueeze(0),
                              candidate=candidate_tensor_env.unsqueeze(0),
                              mask=mask_tensor_env.unsqueeze(0),
                              adj=adj_tensor_env,
                              graph_pool=dag_pool_step)
                
            _, state, reward, done, candidate, mask = env.step(task=int(task_action),
                                                                           device=int(ix_machine_action))
            ep_reward += reward  
            if done:
                break

        log_rewards.append(ep_reward - env.posRewards)
        
    return np.array(log_rewards)


if __name__ == '__main__':
    print("No done.")

