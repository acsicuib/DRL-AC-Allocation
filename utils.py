import numpy as np

""" Get the computational and network times of tasks
The network latency plus service time of a Job is the sum of Tasks belong to Job

Compute the service time (time required / machine's processing time (feat[,0])) 
plus the latency of allocated machine 

Example:
    JobY = {T0, T1,T2}
    timesJobY = 10,5,20
    adjMatrix:
         T0 > T1 > T2
         Tx > T1

    allocations:
          T0 in M0 (execution time: 20, latency: 10)
          T1 in M1 (10,1)
          T2 in M2 (1,5)
          Tx in M0
                
    times:
         T0 =  10/20 + 10  = 10.5      
         T1 =  5/10  + 1+1 = 2.5
         T2 =  20/1  + 5   = 25
    
"""
def getCNTimes(allocations,times,feat,adj):
    idx_machines = np.argmax(allocations, axis=0)
    feat_i = np.take(feat,idx_machines,axis=0)
    # Time initialisation of each task with the service time 
    cntimes = times.reshape(-1)/feat_i[:,0] 
    # Plus latency times by dependencies
    for task in range(allocations.shape[1]):
        taskDependencies = np.where(adj[task,:task])[0]
        # No dependencies, generally first tasks 
        if len(taskDependencies)==0:
            cntimes[task]+=feat_i[:,2][task]
        # Check placement of dependent tasks     
        for td in taskDependencies:
            if idx_machines[td]!=idx_machines[task]: 
                cntimes[task]+=feat_i[:,2][task]
    return cntimes.reshape(times.shape).astype(np.float32)


if __name__ == '__main__':
    from DAG_app_generator import generate_DAG_application

    SEED = 2023
    np.random.seed(SEED)


    n_jobs = 3
    n_machines = 5
    times, adj = generate_DAG_application(n_jobs,3,20)

    # times = np.array([[1, 1 ],[ 2 , 2 ]])
    times = np.array([[1, 1, 1 ],[ 2 , 4, 2 ],[ 4 , 4, 4 ]])
    adj[3,1]=0

    adj[4,0]=-1
    
    print(adj)
    # print(adj.shape)
    print("times")
    print(times)

    HW_options = [2,4,6]
    Cost_options = [1,3]
    latency_options = [5,10] #invertir category max-> min

    feat_labels = ["Processing time","Cost","Lat","Load","Load Penalty"] 
    n_features = len(feat_labels)
    feat_HW = np.random.choice(HW_options,n_machines)
    feat_Cost = np.random.choice(Cost_options,n_machines)
    feat_Lat = np.random.choice(latency_options,n_machines)
    feat_Load = np.repeat([n_features],n_machines) #TODO
    feat_LoadPena = np.zeros(n_machines)  

    feat = np.concatenate((feat_HW,feat_Cost,feat_Lat,feat_Load,feat_LoadPena)).reshape(n_features,n_machines).T
    # assert np.array_equal(m_HW_features,m_fea[:,0])
    feat[0]= np.array([2,-3,1,-3,-3]) # Cloud entity
    feat[-1] = np.array([20,-3,10,-3,-3]) # Cloud entity

    print("HW features: ",feat_labels)
    print(feat)

    allocations = np.zeros((n_machines,n_jobs**2),dtype=np.bool8)
    for jobi in range(times.size):
        allocations[-1,jobi]=True


    # allocations[2,4]=True
    # allocations[-1,4]=False
    allocations[:,3]=False
    allocations[:,4]=False
    allocations[:,5]=False
    allocations[0,3]=True
    allocations[1,4]=True
    allocations[0,5]=True
    print(allocations)

    t = getCNTimes(allocations,times,feat,adj)
    print(t)
    print(np.sum(t,axis=1)) #time by job
    print(np.sum(t)) #time by app