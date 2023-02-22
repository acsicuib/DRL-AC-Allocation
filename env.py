import gym
import numpy as np
from gym.utils import EzPickle
import configs
from utils import getCNTimes
import sys

def descomposeAction(action,number_tasks=configs.n_tasks,number_jobs=configs.n_jobs):
    task = action%number_tasks
    machine = action//number_tasks
    job = task //number_jobs 
    return job, task, machine

class SPP(gym.Env, EzPickle): #Service Placement Problem 
    def __init__(self,
                 number_jobs,
                 number_machines, 
                 number_features):
        EzPickle.__init__(self)
        self.step_count = 0
        
        self.number_jobs = number_jobs
        self.number_tasks = self.number_jobs**2  
        self.number_machines = number_machines+1 # plus the Cloud entity
        self.number_features = number_features
        
        # the task id for first column
        self.first_col = np.arange(0,self.number_tasks,step=int(self.number_jobs))
        # Intermediate tasks for jobs
        self.last_col = np.arange(0,self.number_tasks).reshape(self.number_jobs,self.number_jobs)[:,:-1].reshape((self.number_tasks)-self.number_jobs)
        
        self.partial_sol_sequeence = []
        self.posRewards  = 0

        #NOTE
        self.state_dim = self.number_tasks*3+self.number_machines*2
        self.action_dim = self.number_tasks*self.number_machines


    def done(self):
        if len(self.partial_sol_sequeence) == self.number_tasks:
            return True
        return False

    def reset(self,times,adj,feat):
        self.step_count = 0

        self.allocations = np.zeros((self.number_machines,self.number_tasks),dtype=np.uint8)
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
        
        # Init allocation to the cloud machine [-1]
        for jobi in range(times.size):
            self.allocations[-1,jobi]=1

        self.LBs = getCNTimes(self.allocations,self.times,self.feat_copy,self.adj)

        # Init. features
        # Worst Response Time
        self.max_endTime = np.sum(self.LBs) #dynamic - step() / can change
        self.initQuality = self.max_endTime #static limit

        # self.max_endTime = times.sum()/self.fea_c.min(axis=0)[0] + self.fea_c.max(axis=0)[2]
        # print("MAX_endTime", self.max_endTime)
        self.opIDsOnMchs = np.argmax(self.allocations,axis=0)
        # Return phase
        # Tasks states
        featuresTasks = np.concatenate((self.LBs.reshape(-1, 1)/configs.et_normalize_coef,
                                  self.finished_mark.reshape(-1, 1),
                                  self.opIDsOnMchs.reshape(-1, 1)), axis=1).astype(np.float32)
        
        # Machines states:: Normalize each column by range
        featureMachines = self.feat_copy[:,[0,2]] #Execution time,Latency : shape(machines,features)
        for col in range(2): #TODO fix definitive number of features
            featureMachines[:,col] /= np.abs(featureMachines[:,col]).max()

        # print("FT",featuresTasks.shape)
        # print("FM",featureMachines.shape)
        
        # state = np.concatenate((featuresTasks.reshape(-1),featureMachines.reshape(-1)),dtype=np.float32)

        return self.allocations,(featuresTasks,featureMachines),self.omega,self.mask

    def selectRndMachine(self,candidate=None):
        machine_rnd = np.random.randint(0,self.number_machines)
        return machine_rnd#*self.number_tasks+candidate


    def selectBestLatencyMachine(self,candidate=None):
        # Testing function
        #TODO improve coluymn
        machine_lowest_latency = self.feat_copy[:,2].argmin() # gets the lowest latency machine
        
        return machine_lowest_latency#*self.number_tasks+candidate


    def step(self,task,machine):
    # def step(self,action):
        # task = action%self.number_tasks
        # machine = action//self.number_tasks

        if task not in self.partial_sol_sequeence: # resolved task
            row = task // self.number_jobs #or job
            col = task % self.number_jobs

            self.step_count += 1
            self.finished_mark[row, col] = 1
            self.partial_sol_sequeence.append(task)

            # Update state
            # Task assignment 
            self.allocations[:,task] = 0
            self.allocations[machine,task] = 1
            # allocation machine by task (index)
            self.opIDsOnMchs = np.argmax(self.allocations,axis=0)
            
            # Metrics update
            self.LBs = getCNTimes(self.allocations,self.times,self.feat_copy,self.adj)
                        
            # Omega/Candidate task & Mask/Finished job
            if task in self.last_col:
                self.omega[task // self.number_jobs] += 1
            else:
                self.mask[task // self.number_jobs] = 1

        # Return phase
        # Tasks states
        featuresTasks = np.concatenate((self.LBs.reshape(-1, 1)/configs.et_normalize_coef,
                                  self.finished_mark.reshape(-1, 1),
                                  self.opIDsOnMchs.reshape(-1, 1)), axis=1).astype(np.float32)
        
        # Machines states
        # Machines states:: Normalize each column by range
        #TODO fix definitive number of features
        featureMachines = self.feat_copy[:,[0,2]] #Execution time,Latency : shape(machines,features)

        for col in range(featureMachines.shape[1]):
            featureMachines[:,col] /= np.abs(featureMachines[:,col]).max()

        # state = np.concatenate((featuresTasks.reshape(-1),featureMachines.reshape(-1)),dtype=np.float32)

        # Reward
        reward = - (np.sum(self.LBs) - self.max_endTime)
        if reward == 0: 
            reward = configs.rewardscale #same action/state as the initial maximum
            self.posRewards += reward
        
        #NOTE 
        # RI :: The reward only depends on the differences with the initial allocation time !! NO
        # RII:: thee reward is relative to the next state
        self.max_endTime = np.sum(self.LBs)
        
        return self.allocations, (featuresTasks,featureMachines), reward, self.done(), self.omega, self.mask

if __name__ == '__main__':
    import env_lab
    env_lab.main()
