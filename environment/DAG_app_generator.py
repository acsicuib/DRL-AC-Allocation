import numpy as np

def generate_DAG_application(n_jobs,low,high,degree=0.3):
    n_task = n_jobs**2
    first_col = np.arange(start=0, stop=n_task, step=int(n_jobs))
        
    adj_tasks_self = np.eye(n_task, k=0, dtype=np.float32)
    adj_tasks_dw = np.eye(n_task, k=-1, dtype=np.float32)
    adj_tasks_dw[first_col]=0

    predecessors = adj_tasks_dw.shape[0]
    assert predecessors>2,"Number of tasks should be bigger than 2"
    for task in range(2,predecessors):
        if np.random.rand()<=degree:
            succ = np.random.randint(0,task-1)
            adj_tasks_dw[task][succ]=1
    

    adj = adj_tasks_self + adj_tasks_dw

    times = np.random.randint(low=low, high=high, size=(n_jobs, n_jobs),dtype=np.int32)


    #UNTODO Random dependencies between tasks
    # try:
    #     adj[3][1] = 1
    #     adj[8][1] = 1
    # except IndexError:
    #     print("Warning. Assigning arcs among tasks. TODO")
    # ##
    
    return times,adj


if __name__ == '__main__':
    print("Test generate_DAG_application function")
    n_jobs = 3
    times, adj = generate_DAG_application(n_jobs,3,20)

    print(adj)
    print(adj.shape)
    print("times")
    print(times)

    print(times.sum()) #Time unit to run all program