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
import pickle as pkl

device = torch.device(configs.device)

def main():

    print("Policy Case: ",configs.name)
    print("\t + tasks: ",configs.n_tasks)
    print("\t + devices: ",configs.n_devices)
    print("\t + episodes: ",configs.max_updates)

    #TODO clean old vars
    # torch.manual_seed(configs.torch_seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(configs.torch_seed)
    # np.random.seed(configs.np_seed_train)
    
    number_all_device_features = len(configs.feature_labels) #TODO fix 
    
    env = SPP(number_jobs=configs.n_jobs, number_devices=configs.n_devices,number_features=number_all_device_features) 
 
    # initialize a PPO agent
    ppo_agent = PPO(env.state_dim)
    # path = 'savedModels/{}.pth'.format(str(configs.name) + "_" +str(configs.n_jobs) + '_' + str(configs.n_devices))
    codeW = str(int(configs.rewardWeightTime*100))+str(int(configs.rewardWeightCost*100))
    path = 'savedModels/%s_%s_%s_w%s.pth'%(str(configs.name),
                                            str(configs.n_jobs),
                                            str(configs.n_devices),
                                            codeW
                                            )
    
    if torch.cuda.is_available(): 
        ppo_agent.policy.load_state_dict(torch.load(path)) #EXPERIMENTS FROM GPYU-server
    else:
        ppo_agent.policy.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    # print(ppo_agent.policy)

    dag_pool_step = dag_pool(graph_pool_type=configs.graph_pool_type,
                             batch_size=torch.Size([1, configs.n_tasks, configs.n_tasks]),
                             n_nodes=configs.n_tasks, device=device)
    

    path_dt = 'datasets/dt_TEST_%s_%i_%i.npz'%(configs.name,configs.n_jobs,configs.n_devices)
    dataset = np.load(path_dt)
    dataset = [dataset[key] for key in dataset]
    data = []
    for sample in range(len(dataset[0])):
        data.append((dataset[0][sample],
                     dataset[1][sample],
                     dataset[2][sample],
                     ))

    log = []
    for i, sample  in enumerate(data):
        
        times, adj, feat = sample
        alloc, state, candidate, mask = env.reset(*sample)
        state_ft = state[0]
        state_fm = state[1]
        init_reward = - env.getRewardInit()
        ep_reward = - env.getRewardInit()
        init_time = env.max_endTime
        init_cost = env.max_endCost 
        while True:

            adj_tensor_env = torch.from_numpy(adj).to(device).to_sparse()
            state_ft_tensor_env = torch.from_numpy(state_ft).to(device) 
            state_fm_tensor_env = torch.from_numpy(state_fm).to(device)
            candidate_tensor_env = torch.from_numpy(candidate).to(device)
            mask_tensor_env = torch.from_numpy(mask).to(device)
            
            with torch.no_grad():
                task_action, _, _, _, _, ix_machine_action, _, _, _ = ppo_agent.policy(
                                                            state_ft=state_ft_tensor_env,
                                                            state_fm=state_fm_tensor_env.unsqueeze(0),
                                                            candidate=candidate_tensor_env.unsqueeze(0),
                                                            mask=mask_tensor_env.unsqueeze(0),
                                                            adj=adj_tensor_env,
                                                            graph_pool=dag_pool_step)
                

            alloc, state, reward, done, candidate, mask = env.step(task=int(task_action),
                                                                           device=int(ix_machine_action))
            ep_reward += reward  

            if done: 
                break
        
        
        #TODO Take care log-size in case of large number of epochs 
        log.append([i, env.max_endTime,env.max_endCost,ep_reward])
        print('Sample %i\tTime: %0.2f || %0.2f\t Cost: %.2f || %0.2f \t Reward: %.2f/%.2f'%(
                i + 1,
                init_time, env.max_endTime, 
                init_cost, env.max_endCost,
                ep_reward,init_reward
        ))


        #TODO improve validation process
        # if (i_update + 1) % 100 == 0:#TODO
        # print(env.times)
        # print(env.opIDsOnMchs)
        # print(env.feat_copy)
        # print(env.feat_copy[97])
        # print(env.feat_copy[55])
        # break
    if configs.record_alloc:
        with open('logs/log_eval2_'+ str(configs.name) + "_" + str(configs.n_jobs) + '_' + str(configs.n_devices)+'.pkl', 'wb') as f:
            pickle.dump(log, f)
    
    
    print("Done\n")


if __name__ == '__main__':
    print("Evaluate our policy")
    start_time = datetime.now().replace(microsecond=0)
    print("Start training: ", start_time)
    main()
    end_time = datetime.now().replace(microsecond=0)
    print("Finish training: ", end_time)
    print("Total time: ",(end_time-start_time))
    print("Done policy test.")