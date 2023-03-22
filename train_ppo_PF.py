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
from validation import validate_model


device = torch.device(configs.device)

def main():

    weigthRange = list(range(11))

    midPoint = int(configs.rewardWeightTime*10)+1 #Model [5,5]

    combinationsWeightTC = list(zip(weigthRange,weigthRange[::-1])) #[(0, 9), (1, 8), (2, 7), (3, 6), (4, 5), (5, 4), (6, 3), (7, 2), (8, 1), (9, 0)]
    weightsToTime = combinationsWeightTC[midPoint:] #[(6, 4), (7, 3), (8, 2), (9, 1), (10, 0)]
    weightsToCost = combinationsWeightTC[:midPoint-1][::-1] #[(4, 6), (3, 7), (2, 8), (1, 9), (0, 10)]

    allCombination = [weightsToTime,weightsToCost]
    
    ### Validate date
    path_dt = 'datasets/dt_VALIDATION_%s_%i_%i.npz'%(configs.name,configs.n_jobs,configs.n_devices)
    dataset = np.load(path_dt)
    dataset = [dataset[key] for key in dataset]
    dataVali = []
    for sample in range(len(dataset[0])):
        dataVali.append((dataset[0][sample],
                    dataset[1][sample],
                    dataset[2][sample],
                    ))
    print("Loading Validation dataset, len: %i"%len(dataVali)) 
    
    
    for weigths in allCombination:
        base_model_code = "55"

        for e,(wt,wc) in enumerate(weigths):
            # print(wt/10.)
            # print(wc/10.)
            # print(".")
            configs.rewardWeightTime = wt/10.
            configs.rewardWeightCost = wc/10.
            print(configs.rewardWeightCost)
            print(configs.rewardWeightTime)

            
            codeW = str(int(configs.rewardWeightTime*10))+str(int(configs.rewardWeightCost*10))
            
            
            print("Training model T %f  C %f"%(configs.rewardWeightTime,configs.rewardWeightCost))
            

            torch.manual_seed(configs.torch_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(configs.torch_seed)
            np.random.seed(configs.np_seed_train)


            number_all_device_features = len(configs.feature_labels) #TODO fix 
            envs = [SPP(number_jobs=configs.n_jobs, number_devices=configs.n_devices,number_features=number_all_device_features) for _ in range(configs.num_envs)]
            
            memories = [Memory() for _ in range(configs.num_envs)]

            # initialize a PPO agent
            ppo_agent = PPO(envs[0].state_dim)

            
            print("Loading previous Model code: ",base_model_code)

            path = 'savedModels/%s_%s_%s_w%s.pth'%(str(configs.name),
                                            str(configs.n_jobs),
                                            str(configs.n_devices),
                                            base_model_code
                                            )
            
            if torch.cuda.is_available(): 
                ppo_agent.policy.load_state_dict(torch.load(path)) #EXPERIMENTS FROM GPYU-server
                # ppo_agent.policy_old.load_state_dict(torch.load(path)) #EXPERIMENTS FROM GPYU-server
            else:
                ppo_agent.policy.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
                # ppo_agent.policy_old.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
            ppo_agent.fromModel()
            # ppo_agent.policy.load_state_dict(torch.load(path)) #TODO policy OLD


            # print(ppo_agent.policy)

            dag_pool_step = dag_pool(graph_pool_type=configs.graph_pool_type,
                                    batch_size=torch.Size([1, configs.n_tasks, configs.n_tasks]),
                                    n_nodes=configs.n_tasks, device=device)
            

            # training loop
            log = []
            logAlloc = []
            validation_log = []
            record_reward_valid = 10000000
            
            for i_update in range(configs.max_updates):
                  #TODO clean vars -> state 
                ep_rewards = np.zeros(configs.num_envs)
                init_rewards = np.zeros(configs.num_envs)
                init_times = np.zeros(configs.num_envs)
                init_costs = np.zeros(configs.num_envs)
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
                    init_times[i] = env.max_endTime
                    init_costs[i] = env.max_endCost
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
                                        logAlloc.append([i,
                                        envs[i].opIDsOnMchs.tolist(), # allocations
                                        envs[i].feat_copy[envs[i].opIDsOnMchs][:,0].tolist(), # Speed
                                        envs[i].feat_copy[envs[i].opIDsOnMchs][:,2].tolist(), # Latency
                                        envs[i].feat_copy[envs[i].opIDsOnMchs][:,1].tolist()  # Cost
                                        ])

                time_all_env, cost_all_env = [], []
                for j in range(configs.num_envs): # Makespan
                    ep_rewards[j] -= envs[j].posRewards # same actions/states as the initial maximum goal state
                    time_all_env.append(envs[j].max_endTime)
                    cost_all_env.append(envs[j].max_endCost)
                time_all_env = np.array(time_all_env) 
                cost_all_env = np.array(cost_all_env) 
                # update PPO agent         
                loss, v_loss  = ppo_agent.update(memories)        

                
                for memory in memories:
                    memory.clear_memory()
            
                mean_rewards_all_env = sum(ep_rewards) / len(ep_rewards)
                mean_all_init_rewards =  init_rewards.mean()

                #TODO Take care log-size in case of large number of epochs 
                log.append([i_update, mean_rewards_all_env,v_loss,mean_all_init_rewards,init_times.mean(),time_all_env.mean(),init_costs.mean(),cost_all_env.mean()])
                print('Episode {} Last reward: {:.2f}\t Mean_Vloss: {:.8f}\t Init reward: {:.2f}\t Init Time: {:.2f}\t Time: {:.2f}\n Init Cost: {:.2f}\t Cost: {:.2f}'.
                    format(i_update + 1, mean_rewards_all_env, v_loss, mean_all_init_rewards,init_times.mean(),time_all_env.mean(),init_costs.mean(),cost_all_env.mean()))
                

                if (i_update + 1) % 10 == 0: #TODO return previous if
                    avg_reward_valid = - validate_model(dataVali, ppo_agent.policy).mean() # return rewards from validate dataset
                    validation_log.append(avg_reward_valid)
                    if avg_reward_valid < record_reward_valid:
                        print("\t Storage the model %i - code: %s"%(i_update+1,codeW))
                        torch.save(ppo_agent.policy.state_dict(), 'savedModels/%s_%s_%s_w%s.pth'%(str(configs.name),
                                                                                                    str(configs.n_jobs),
                                                                                                    str(configs.n_devices),
                                                                                                    codeW
                                                                                                    ))

                        record_reward_valid = avg_reward_valid
                        
                        file_writing_obj1 = open(
                                'logs/vali_' + str(configs.name) +"_w" + codeW + '.txt', 'w')
                        file_writing_obj1.write(str(validation_log))
                        file_writing_obj1.close()

                # t5 = time.time()


            #Store the logs
            if configs.record_ppo:
                with open('logs/log_ppo_' + str(configs.name) + "_w" + codeW +'.pkl', 'wb') as f:
                    pickle.dump(log, f)
            
            if configs.record_alloc:
                with open('logs/log_ppo_alloc_'+ str(configs.name) + "_w" + codeW +'.pkl', 'wb') as f:
                    pickle.dump(logAlloc, f)
            
            
            print("Done: _w%s\n"%base_model_code)
            base_model_code = str(int(configs.rewardWeightTime*10))+str(int(configs.rewardWeightCost*10))
            # break
        
        # break
        print(".. Changing direction of weigths")




if __name__ == '__main__':
    print("TRAINING PF policy")
    start_time = datetime.now().replace(microsecond=0)
    print("Start training: ", start_time)
    main()
    end_time = datetime.now().replace(microsecond=0)
    print("Finish training: ", end_time)
    print("Total time: ",(end_time-start_time))
    

    file_writing_obj1 = open(
        'logs/exec_train_ppo_PF_time_' + str(configs.name) +"_" + str(configs.n_jobs) + '_' + str(configs.n_devices) + '.txt', 'w')
    file_writing_obj1.write(str((end_time-start_time)))
    file_writing_obj1.close()

    print("Done policy test.")