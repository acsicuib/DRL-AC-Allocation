import gym
import numpy as np
from gym.utils import EzPickle
from parameters import configs
from environment.utils import getCNTimes, getCNCosts
import sys

# def descomposeAction(action,number_tasks=configs.n_tasks,number_jobs=configs.n_jobs):
#     task = action%number_tasks
#     device = action//number_tasks
#     job = task //number_jobs 
#     return job, task, device

class SPP(gym.Env, EzPickle): #Service Placement Problem 
    def __init__(self,
                 number_jobs,
                 number_devices, 
                 number_features):
        EzPickle.__init__(self)
        self.step_count = 0
        
        self.number_jobs = number_jobs
        self.number_tasks = self.number_jobs**2  
        self.number_devices = number_devices+1 # plus the Cloud entity
        self.number_features = number_features
        
        # the task id for first column
        self.first_col = np.arange(0,self.number_tasks,step=int(self.number_jobs))
        # Intermediate tasks for jobs
        self.last_col = np.arange(0,self.number_tasks).reshape(self.number_jobs,self.number_jobs)[:,:-1].reshape((self.number_tasks)-self.number_jobs)
        
        self.partial_sol_sequeence = []
        self.posRewards  = 0

        #NOTE
        self.state_dim = self.number_tasks*3+self.number_devices*2
        self.action_dim = self.number_tasks*self.number_devices


    def done(self):
        if len(self.partial_sol_sequeence) == self.number_tasks:
            return True
        return False

    def reset(self,times,adj,feat):
        self.step_count = 0

        self.allocations = np.zeros((self.number_devices,self.number_tasks),dtype=np.uint8)
        self.adj = np.copy(adj)
        self.feat_copy = np.copy(feat)
        self.times = np.copy(times)

        # record action history
        self.partial_sol_sequeence = []
        self.flags = []
        self.posRewards = 0

        # candidate tasks        
        self.omega = self.first_col.astype(np.int64)
        # print("Omega")
        # print(self.omega)
        
        # initialize mask
        self.mask = np.full(shape=self.number_jobs, fill_value=0, dtype=bool)
        # print("Mask")
        # print(self.mask)

        self.finished_mark = np.zeros_like(self.times, dtype=np.uint8)
        # print(self.finished_mark)
        
        # Init allocation to the cloud Device [-1]
        for jobi in range(times.size):
            self.allocations[-1,jobi]=1

        self.LBs = getCNTimes(self.allocations,self.times,self.feat_copy,self.adj)
        self.Costs = getCNCosts(self.allocations,self.feat_copy)

        # Init. features
        # Worst Response Time
        self.max_endTime = np.sum(self.LBs) #dynamic - step() / can change
        self.max_endCost = np.sum(self.Costs)
        self.initQuality = self.getRewardInit() #self.max_endTime #static limit

        # self.max_endTime = times.sum()/self.fea_c.min(axis=0)[0] + self.fea_c.max(axis=0)[2]
        # print("MAX_endTime", self.max_endTime)
        self.opIDsOnMchs = np.argmax(self.allocations,axis=0)
        # Return phase
        # Tasks states
        featuresTasks = np.concatenate((self.LBs.reshape(-1, 1)/configs.et_normalize_coef,
                                        self.Costs.reshape(-1, 1)/configs.et_normalize_coef_cost,
                                        self.finished_mark.reshape(-1, 1),
                                        self.opIDsOnMchs.reshape(-1, 1)), axis=1).astype(np.float32)
        
        # Device states:: Normalize each column by range
        featureDevices = self.feat_copy[:,[0,1,2]] #Execution time,Latency : shape(Device,features)
        for col in range(featureDevices.shape[1]): 
            featureDevices[:,col] /= np.abs(featureDevices[:,col]).max()


        return self.allocations,(featuresTasks,featureDevices),self.omega,self.mask

    def selectRndDevice(self):
        rnd_device = np.random.randint(0,self.number_devices)
        return rnd_device#*self.number_tasks+candidate

    def selectCloudDevice(self):
        return self.number_devices-1
    

    def selectBestLatencyDevice(self):
        #TODO improve selection of columns: value=2
        device_lowest_latency = self.feat_copy[:,2].argmin() # gets the lowest latency device
        return device_lowest_latency

    def selectBestCostDevice(self):
        #TODO improve selection of columns: value=2
        device_lowest_cost = self.feat_copy[:,1].argmin() # gets the lowest latency device
        return device_lowest_cost
    
    def selectDevicePriorizingLat(self):
        #TODO improve selection of columns: value=2
        lowLatencyValue = self.feat_copy[:,2].min() # gets the lowest latency device
        devices = self.feat_copy[self.feat_copy[:,2]==lowLatencyValue]
        device_lowest_cost = devices[:,1].argmin()
        result = np.where(np.all(self.feat_copy == devices[device_lowest_cost],axis=1))[0][0]
        return result

    def selectDevicePriorizingCost(self):
        #TODO improve selection of columns: value=2
        costValue = self.feat_copy[:,1].min() # gets the lowest latency device
        devices = self.feat_copy[self.feat_copy[:,1]==costValue]
        device_lowest_lat = devices[:,2].argmin()
        result = np.where(np.all(self.feat_copy == devices[device_lowest_lat],axis=1))[0][0]
        return result
    

    def getRewardInit(self):
        return configs.rewardWeightTime*(self.max_endTime) + configs.rewardWeightCost*(self.max_endCost)
    
    # def getCurrentTime(self):
    #     return np.sum(self.LBs) #== max_endTime
    
    # def getCurrentCost(self): 
    #     return np.sum(self.Costs) #== man_endCost
    
    def step(self,task, device):
        if task not in self.partial_sol_sequeence: # resolved task
            row = task // self.number_jobs #or job
            col = task % self.number_jobs

            self.step_count += 1
            self.finished_mark[row, col] = 1
            self.partial_sol_sequeence.append(task)

            # Update state
            # Task assignment 
            self.allocations[:,task] = 0
            self.allocations[device,task] = 1
            # allocation device by task (index)
            self.opIDsOnMchs = np.argmax(self.allocations,axis=0)
            
            # Update metrics
            self.LBs = getCNTimes(self.allocations,self.times,self.feat_copy,self.adj)
            self.Costs = getCNCosts(self.allocations,self.feat_copy) 

            # Omega/Candidate task & Mask/Finished job
            if task in self.last_col:
                self.omega[task // self.number_jobs] += 1
            else:
                self.mask[task // self.number_jobs] = 1

        # Return phase
        # Tasks states
        featuresTasks = np.concatenate((self.LBs.reshape(-1, 1)/configs.et_normalize_coef,
                                        self.Costs.reshape(-1, 1)/configs.et_normalize_coef_cost,
                                        self.finished_mark.reshape(-1, 1),
                                        self.opIDsOnMchs.reshape(-1, 1)), axis=1).astype(np.float32)
        
        # Device states
        #TODO fix definitive number of features
        featureDevices = self.feat_copy[:,[0,1,2]] #Execution time, COST , Latency : shape(device,features)

        for col in range(featureDevices.shape[1]):
            featureDevices[:,col] /= np.abs(featureDevices[:,col]).max()

        # Reward
        reward = - (configs.rewardWeightTime*(np.sum(self.LBs) - self.max_endTime) + ((configs.rewardWeightCost)*(np.sum(self.Costs)-self.max_endCost)))
        if reward == 0: 
            reward = configs.rewardscale #same action/state as the initial maximum
            self.posRewards += reward

        self.max_endTime = np.sum(self.LBs)
        self.max_endCost = np.sum(self.Costs)
        
        return self.allocations, (featuresTasks,featureDevices), reward, self.done(), self.omega, self.mask

if __name__ == '__main__':
    print("Run file: /test_env.py")
