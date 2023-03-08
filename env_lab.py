from parameters import configs
from env import SPP
from instance_generator import one_instance_gen
import numpy as np
import time


def main():
    SEED = 2023
    np.random.seed(SEED)

    n_jobs = configs.n_jobs 
    n_devices = configs.n_devices
    n_features = len(configs.feature_labels)

    times, adj, feat = one_instance_gen(n_jobs,n_devices,configs.cloud_features,configs.DAG_rand_dependencies_factor)
    print("HW features: ",feat)
    print(feat)

    t1 = time.time()

    # Environment
    env = SPP(number_jobs=n_jobs,number_devices=n_devices,number_features=n_features)

    print("Reset environment")
    alloc, state, omega, mask = env.reset(times,adj,feat)
    print("LBs")
    print(env.LBs)
    
    print("Initial time:")
    print(np.sum(env.LBs))
    print(env.max_endTime) 

    print("posRewards")
    print(env.posRewards) 

    print("Allocations")
    print(alloc)
    rewards = [-env.max_endTime]
    print("*"*30)

    while True:
        print("\tEnv.step: ",env.step_count)
        print("*"*30)
        ix_job = np.random.choice(len(omega[~mask]))
        candidate_task = omega[~mask][ix_job]
        print("Candidate_job: ",candidate_task)
        device = env.selectRndDevice()
        print('Action:', device)

        alloc, state, reward, done, omega, mask = env.step(candidate_task,device)
        rewards.append(reward)
        print("Post alloc by task")
        print(env.opIDsOnMchs)
        # print("post FEAT")
        # print(featTasks)
        # print(featTasks.shape)
        
        # print("post OMEGA\n",omega)
        # print("post Mask\n",mask)
        # print("post Alloc")
        # print(env.allocations)
        print('post LBs:\n', env.LBs)
        print("post reward:\n", reward)
        print("post posReward:\n", env.posRewards)

        # input("Press Enter to continue...") #debug
        
        if env.done():
            break

    print("DONE")
    print("+"*30)
    makespan = sum(rewards) - env.posRewards
    t2 = time.time()

    print("MakeSpan")
    print(makespan)
    print("LB")
    print(env.LBs)
    print(np.sum(env.LBs))
    # print("time")
    # print(t2 - t1)
    # # np.save('sol', env.opIDsOnMchs // n_m)
    # np.save('jobSequence', env.opIDsOnMchs)
    # np.save('testData', data)
    print("Allocations by task")
    print(env.opIDsOnMchs)
    print("Number of steps")
    print(env.step_count)
    print("Rewards")
    print(rewards)
    print(sum(rewards)) # equivalent to: -1 * np.sum(env.LBs)


if __name__ == '__main__':
    main()