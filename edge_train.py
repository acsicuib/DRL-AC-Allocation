import os
import glob
import time
import sys

import pickle
import numpy as np
from datetime import datetime

import configs
from env import *
from instance_generator import one_instance_gen

###
# Reward TEST in the environment when it chooses the best machine in terms of latency 
###

def main():
    np.random.seed(configs.np_seed_train)
 
    envs = [SPP(number_jobs=configs.n_jobs, number_machines=configs.n_machines,number_features=configs.n_feat) for _ in range(configs.num_envs)]
    data_generator = one_instance_gen
  
    # training loop
    log = []
    for i_update in range(configs.max_updates):

        ep_rewards = np.zeros(configs.num_envs)
        init_rewards = np.zeros(configs.num_envs)
        candidate_envs = []
        mask_envs = []
        
        # Init all the environments
        for i, env in enumerate(envs):
            _, _, candidate, mask = env.reset(*data_generator(n_jobs=configs.n_jobs, n_machines=configs.n_machines))

            candidate_envs.append(candidate)
            mask_envs.append(mask)
            ep_rewards[i] = - env.initQuality
            # print("\tR%i: %f "%(0,ep_rewards[i]))
            init_rewards[i] = - env.initQuality

        steps = 0
        while True:
            action_envs = []
            steps+=1

            for i in range(configs.num_envs):
                # V0. select rnd action. Version amigable
                ix_job = np.random.choice(len(candidate_envs[i][~mask_envs[i]]))
                candidate_task = candidate_envs[i][~mask_envs[i]][ix_job]
                machine = envs[i].selectBestLatencyMachine()
    
                # V1. 1º Cualquiera sin los candidatos y posteriormente aplicando candidatos (modelo ppo_train.py)  # V1. 1º Cualquier sin los candidatos 
                ### NO FUNCIONA - hay que aplicar el candidate
                # action = np.random.randint(0,envs[i].action_dim)

                action_envs.append((candidate_task,machine))

            candidate_envs = []
            mask_envs = []
    
            # Saving episode data
            for i in range(configs.num_envs):
                # print("Task:",action_envs[i][0])
                _, _, reward, done, candidate, mask = envs[i].step(task=action_envs[i][0],
                                                                machine=action_envs[i][1])

                candidate_envs.append(candidate)
                mask_envs.append(mask)
                # print("MASK ",mask)
                # print("DONE ",done)
                
                ep_rewards[i] += reward
                # print("\tR%i\t %f \t %f \t %f \t %f"%(steps,reward,envs[i].max_endTime,ep_rewards[i],np.sum(envs[i].LBs)))

            if envs[0].done(): #all environments are DONE (same number of tasks)
                assert steps == envs[0].step_count
                break
            
        for j in range(configs.num_envs):  #Get the last one
            ep_rewards[j] -= envs[j].posRewards # same actions/states as the initial maximum goal state
        
        # for i in range(configs.num_envs):
        #     print("%i ENV allocations: %i "%(i_update + 1,i))
        #     print(envs[i].opIDsOnMchs)
        #     print(envs[i].feat_copy)

        # ep_rewards represents the computational and network time for the current allocation
        mean_rewards_all_env = ep_rewards.mean() # mean of the c-n time 
        mean_all_init_rewards =  init_rewards.mean()
        log.append([i_update, mean_rewards_all_env, mean_all_init_rewards])

        print('Episode {}\t Last reward: {:.2f} \t Init reward: {:.2f}'.format(i_update + 1, mean_rewards_all_env, mean_all_init_rewards))
    
    with open('trainlogs/log_edge_train_' + str(configs.n_jobs) + '_' + str(configs.n_machines) + '_' + str(configs.task_time_low) + '_' + str(configs.task_time_high)+'.pkl', 'wb') as f:
        pickle.dump(log, f)


if __name__ == '__main__':
    print("Random Experiment")
    start_time = datetime.now().replace(microsecond=0)
    print("Started training: ", start_time)
    print("="*30)
    main()
    end_time = datetime.now().replace(microsecond=0)
    print("Finish training: ", end_time)
    print("Total time: ",(end_time-start_time))
    print("Done.")