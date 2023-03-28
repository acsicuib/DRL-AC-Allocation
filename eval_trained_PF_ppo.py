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

    torch.manual_seed(configs.torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(configs.torch_seed)
    np.random.seed(configs.np_seed_train)

    weigthRange = list(np.arange(0,11,2.5))
    combinationsWeightTC = list(zip(weigthRange,weigthRange[::-1])) #[(0, 10), (1, 9), (2, 8), (3, 7), (4, 6), (5, 5), (6, 4), (7, 3), (8, 2), (9, 1), (10, 0)]
    combinationsWeightTC = [(0.0, 10.0), (5.0, 5.0)]
    # combinationsWeightTC = [(0.0, 10.0), (2.5, 7.5), (5.0, 5.0), (7.5, 2.5), (10.0, 0.0)]
    print(combinationsWeightTC)

    ## TEST Dataset
    path_dt = 'datasets/dt_TEST_%s_%i_%i.npz'%(configs.name,configs.n_jobs,configs.n_devices)
    # path_dt = 'datasets/dt_VALIDATION_%s_%i_%i.npz'%(configs.name,configs.n_jobs,configs.n_devices)
    dataset = np.load(path_dt)
    dataset = [dataset[key] for key in dataset]
    data = []
    for sample in range(len(dataset[0])):
        data.append((dataset[0][sample],
                     dataset[1][sample],
                     dataset[2][sample],
                     ))

    print("Loading Test dataset, len: %i"%len(data)) 
    
    number_all_device_features = len(configs.feature_labels) #TODO fix

    log = []

    for e,(wt,wc) in enumerate(combinationsWeightTC[::-1]):
        configs.rewardWeightTime = wt/10.
        configs.rewardWeightCost = wc/10.
        print("T",configs.rewardWeightTime)
        print("C",configs.rewardWeightCost)
        

        codeW = str(int(configs.rewardWeightTime*100))+str(int(configs.rewardWeightCost*100))
        # codeW ="0100"
        # configs.rewardWeightTime = 0.0
        # configs.rewardWeightCost = 1.0
        
        print("Model combination: _w",codeW)
        


        env = SPP(number_jobs=configs.n_jobs, number_devices=configs.n_devices,number_features=number_all_device_features) 
 
        # initialize a PPO agent & loading the model
        ppo_agent = PPO(env.state_dim)
        
        path = 'savedModels/%s_%s_%s_w%s.pth'%(str(configs.name),
                                            str(configs.n_jobs),
                                            str(configs.n_devices),
                                            codeW
                                            )
        print(" Loading path : ",path)

        if torch.cuda.is_available(): 
            ppo_agent.policy.load_state_dict(torch.load(path)) #EXPERIMENTS FROM GPYU-server
        else:
            ppo_agent.policy.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

        ppo_agent.fromModel()
    

        dag_pool_step = dag_pool(graph_pool_type=configs.graph_pool_type,
                                batch_size=torch.Size([1, configs.n_tasks, configs.n_tasks]),
                                n_nodes=configs.n_tasks, device=device)
    

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
            

            log.append([codeW, i, env.max_endTime,env.max_endCost,ep_reward,init_time,init_cost])
            print('Model: %s\t Sample %i\tTime: %0.2f || %0.2f\t Cost: %.2f || %0.2f \t Reward: %.2f/%.2f'%(
                    codeW,
                    i + 1,
                    init_time, env.max_endTime, 
                    init_cost, env.max_endCost,
                    ep_reward,init_reward
            ))

        # break
    if configs.record_alloc:
        with open('logs/log_eval_PF_'+ str(configs.name) + "_" + str(configs.n_jobs) + '_' + str(configs.n_devices)+'.pkl', 'wb') as f:
            pickle.dump(log, f)
    
    
    print("Done\n")


if __name__ == '__main__':
    print("Evaluate our policy with PF models")
    start_time = datetime.now().replace(microsecond=0)
    print("Start training: ", start_time)
    main()
    end_time = datetime.now().replace(microsecond=0)
    print("Finish training: ", end_time)
    print("Total time: ",(end_time-start_time))
    print("Done policy test.")