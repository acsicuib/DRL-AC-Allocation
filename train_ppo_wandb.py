import os
import sys
import glob
import time
import torch
import numpy as np
import pickle
from datetime import datetime

from parameters import configs
from environment.env import *
from policy import PPO, Memory
from instance_generator import one_instance_gen
from models.dag_aggregate import dag_pool

import wandb

# Example sweep configuration
count = 20
sweep_configuration = {
    'method': 'bayes',
    'name': 'sweep-v2',
    'metric': {
        'goal': 'maximize', 
        'name': 'reward'
        },
    'parameters': {
        # 'batch_size': {'values': [16, 32, 64]},
        'num_layers': {'values': [1, 2, 3, 4, 5]},
        'num_mlp_layers_actor': {'values': [1, 2, 3, 4, 5]},
        'num_mlp_layers_critic': {'values': [1, 2, 3, 4, 5]},
        'k_epochs': {'values': [1, 2, 3]},
        'eps_clip': {'max': 0.5, 'min': 0.01},
        'lr_agent': {'max': 0.1, 'min': 0.0001},
        'lr_critic': {'max': 0.1, 'min': 0.0001},
        'entloss_coef': {'max': 0.5, 'min': 0.001},
        'vloss_coef': {'values': [1, 2, 3]},
        'ploss_coef': {'values': [1, 2, 3]}
     }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="project-DRL-AC")

device = torch.device(configs.device)

def main():
    run = wandb.init()

    print("Policy Case: ",configs.name)
    print("\t + tasks: ",configs.n_tasks)
    print("\t + devices: ",configs.n_devices)
    print("\t + episodes: ",configs.max_updates)


    configs.lr_agent = wandb.config.lr_agent
    configs.lr_critic = wandb.config.lr_critic
    configs.num_layers = wandb.config.num_layers
    configs.num_mlp_layers_actor = wandb.config.num_mlp_layers_actor
    configs.num_mlp_layers_critic = wandb.config.num_mlp_layers_critic
    configs.k_epochs = wandb.config.k_epochs
    configs.eps_clip = wandb.config.eps_clip
    configs.lr_agent = wandb.config.lr_agent
    configs.lr_critic = wandb.config.lr_critic
    configs.entloss_coef = wandb.config.entloss_coef
    configs.vloss_coef = wandb.config.vloss_coef
    configs.ploss_coef = wandb.config.ploss_coef
    
    


    #TODO clean old vars
    torch.manual_seed(configs.torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(configs.torch_seed)
    np.random.seed(configs.np_seed_train)
    
    number_all_device_features = len(configs.feature_labels) #TODO fix 
    envs = [SPP(number_jobs=configs.n_jobs, number_devices=configs.n_devices,number_features=number_all_device_features) for _ in range(configs.num_envs)]
 
    memories = [Memory() for _ in range(configs.num_envs)]

    # initialize a PPO agent
    ppo_agent = PPO(envs[0].state_dim)
    # print(ppo_agent.policy)

    dag_pool_step = dag_pool(graph_pool_type=configs.graph_pool_type,
                             batch_size=torch.Size([1, configs.n_tasks, configs.n_tasks]),
                             n_nodes=configs.n_tasks, device=device)
    
    # training loop
    log = []
    logAlloc = []
    for i_update in range(configs.max_updates):
        
        #TODO clean vars -> state 
        ep_rewards = np.zeros(configs.num_envs)
        init_rewards = np.zeros(configs.num_envs)
        # alloc_envs = []
        state_ft_envs,state_fm_envs= [],[]
        candidate_envs = []
        mask_envs = []
        adj_envs = []


        # Init all the environments
        for i, env in enumerate(envs):
            alloc, state, candidate, mask = env.reset(*one_instance_gen(n_jobs=configs.n_jobs, n_devices=configs.n_devices,cloud_features=configs.cloud_features, dependency_degree=configs.DAG_rand_dependencies_factor))
            adj_envs.append(env.adj)
            # alloc_envs.append(alloc)
            state_ft_envs.append(state[0])
            state_fm_envs.append(state[1])
            candidate_envs.append(candidate)
            mask_envs.append(mask)
            ep_rewards[i] = - env.initQuality
            init_rewards[i] = - env.initQuality

        steps = 0    
        while True:
            steps+=1

            adj_tensor_envs = [torch.from_numpy(np.copy(adj)).to(device).to_sparse() for adj in adj_envs]
            # alloc_tensor_envs = [torch.from_numpy(np.copy(alloc)).to(device) for alloc in alloc_envs]
            state_ft_tensor_envs = [torch.from_numpy(np.copy(st)).to(device) for st in state_ft_envs]
            state_fm_tensor_envs = [torch.from_numpy(np.copy(st)).to(device) for st in state_fm_envs]
            candidate_tensor_envs = [torch.from_numpy(np.copy(candidate)).to(device) for candidate in candidate_envs]
            mask_tensor_envs = [torch.from_numpy(np.copy(mask)).to(device) for mask in mask_envs]
            
            with torch.no_grad():
                task_action_envs,m_action_envs  = [],[]
                task_idx_envs, m_idx_envs = [],[]

                for i in range(configs.num_envs):
                 # select action with policy
                    # state = torch.cat((feat_task_tensor_envs[i].reshape(-1),feat_mach_tensor_envs[i].reshape(-1)))
                    # state = state.type(torch.float)

                    task_action, ix_task_action, _, _, logProb, ix_machine_action, _, _, logProb_m = ppo_agent.policy_old(
                                                                state_ft=state_ft_tensor_envs[i],
                                                                state_fm=state_fm_tensor_envs[i].unsqueeze(0),
                                                                candidate=candidate_tensor_envs[i].unsqueeze(0),
                                                                mask=mask_tensor_envs[i].unsqueeze(0),
                                                                adj=adj_tensor_envs[i],
                                                                graph_pool=dag_pool_step)
                 
                    # print(action)
                    # print(a_idx)
                    task_action_envs.append(task_action)
                    task_idx_envs.append(ix_task_action) 
                    m_idx_envs.append(ix_machine_action)

                    memories[i].logprobs.append(logProb)
                    memories[i].logprobs_m.append(logProb_m)
                    # m_idx_envs.append(log_machprob)

            # alloc_envs = []
            state_ft_envs = []
            state_fm_envs = []
            
            # featT_envs = []
            # featM_envs = []
            candidate_envs = []
            mask_envs = []    

            # Saving episode data
            for i in range(configs.num_envs):
                memories[i].adj_mb.append(adj_tensor_envs[i]) #TODO Purge memories
                # memories[i].alloc_mb.append(alloc_tensor_envs[i])
                memories[i].state_ft.append(state_ft_tensor_envs[i])
                memories[i].state_fm.append(state_fm_tensor_envs[i])
                # memories[i].featTask.append(feat_task_tensor_envs[i])
                # memories[i].featMach.append(feat_mach_tensor_envs[i])
                memories[i].candidate_mb.append(candidate_tensor_envs[i])
                memories[i].mask_mb.append(mask_tensor_envs[i])
                memories[i].a_mb.append(task_idx_envs[i]) #clean both vars.
                memories[i].am_mb.append(m_idx_envs[i]) #clean both vars.
                

                alloc, state, reward, done, candidate, mask = envs[i].step(task=int(task_action_envs[i]),
                                                                           device=int(m_idx_envs[i]))
                

                # alloc_envs.append(alloc)
                state_ft_envs.append(state[0])
                state_fm_envs.append(state[1])
                # featT_envs.append(featTasks)
                # featM_envs.append(featMachs)
                candidate_envs.append(candidate)
                mask_envs.append(mask)

                ep_rewards[i] += reward
                memories[i].reward_mb.append(reward)
                memories[i].done_mb.append(done)
            
            if envs[0].done():  #all environments are DONE (same number of tasks)
                assert steps == envs[0].step_count
                break

        
        # if i_update in [0,5,10,20]:
        if i_update in configs.record_alloc_episodes:
            # print("Final placement: ",i_update)
            # print(" -"*30)
            for i in range(configs.num_envs): # Makespan
                # print(i,envs[i].opIDsOnMchs,envs[i].feat_copy[envs[i].opIDsOnMchs][:,0],envs[i].feat_copy[envs[i].opIDsOnMchs][:,2])
                logAlloc.append([i,envs[i].opIDsOnMchs.tolist(),envs[i].feat_copy[envs[i].opIDsOnMchs][:,0].tolist(),envs[i].feat_copy[envs[i].opIDsOnMchs][:,2].tolist()])

        for j in range(configs.num_envs): # Makespan
            ep_rewards[j] -= envs[j].posRewards # same actions/states as the initial maximum goal state
                
        # update PPO agent         
        loss, v_loss  = ppo_agent.update(memories)        

        
        for memory in memories:
            memory.clear_memory()
       
        mean_rewards_all_env = sum(ep_rewards) / len(ep_rewards)
        mean_all_init_rewards =  init_rewards.mean()
        log.append([i_update, mean_rewards_all_env,v_loss,mean_all_init_rewards])
        print('Episode {}\t Last reward: {:.2f}\t Mean_Vloss: {:.8f}\t Init reward: {:.2f}'.format(i_update + 1, mean_rewards_all_env, v_loss, mean_all_init_rewards))
        wandb.log({"epoch":i_update + 1,"v_loss": v_loss, "loss": loss, "reward":mean_rewards_all_env, "init_reward":mean_all_init_rewards})
        ## DEBUG with out PPO Agent -. 
        # mean_rewards_all_env = ep_rewards.mean() # mean of the c-n time 
        # mean_all_init_rewards =  init_rewards.mean()
        # log.append([i_update, mean_rewards_all_env, mean_all_init_rewards])
        # print('Episode {}\t Last reward: {:.2f} \t Init reward: {:.2f}'.format(i_update + 1, mean_rewards_all_env, mean_all_init_rewards))

    #Store the logs
    if configs.record_ppo:
        with open('logs/log_ppo_'  + str(configs.name) + "_" + str(configs.n_jobs) + '_' + str(configs.n_devices)+'.pkl', 'wb') as f:
            pickle.dump(log, f)
    
    if configs.record_alloc:
        with open('logs/log_ppo_alloc_'+ str(configs.name) + "_" + str(configs.n_jobs) + '_' + str(configs.n_devices)+'.pkl', 'wb') as f:
            pickle.dump(logAlloc, f)
    
    wandb.finish()
    print("Done\n")


if __name__ == '__main__':
    print("TRAINING our policy in wandb plataform")
    start_time = datetime.now().replace(microsecond=0)
    print("Start training: ", start_time)
    wandb.agent(sweep_id, function=main, count=count)
    end_time = datetime.now().replace(microsecond=0)
    print("Finish training: ", end_time)
    print("Total time: ",(end_time-start_time))
    print("Done policy test.")