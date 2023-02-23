import numpy as np
from DAG_app_generator import generate_DAG_application
import configs


def one_instance_gen(n_jobs,n_machines,cloud_features=configs.cloud_features,dependency_degree=configs.DAG_rand_dependencies_factor):

    times, adj = generate_DAG_application(n_jobs,configs.task_time_low,configs.task_time_high,degree=dependency_degree)
    n_features = len(configs.feat_labels)
    
    feat_Speed = np.random.choice(configs.cpu_speed,n_machines)
    feat_Cost = np.random.choice(configs.cost_options,n_machines)
    feat_Lat = np.random.choice(configs.latency_options,n_machines)
    feat_Load = np.repeat([n_features],n_machines) #TODO HW features
    # feat_LoadPena = np.zeros(n_machines)  
    # feat = np.concatenate((feat_HW,feat_Cost,feat_Lat,feat_Load,feat_LoadPena)).reshape(n_features,n_machines).T
    
    feat = np.concatenate((feat_Speed,feat_Cost,feat_Lat,feat_Load)).reshape(n_features,n_machines).T
    
    # last machine represents the cloud entity
    feat = np.vstack((feat,cloud_features)) #Cloud is always the same
    

    
    # ! torch.float
    feat = feat.astype(np.float32)
    
    return times, adj, feat

if __name__ == '__main__':
    n_jobs = 3
    n_machines = 5 # In total = 5+1, cloud entity
    cloud_features = [20,2,4,0]
    for i in range(10):
        times, adj,feat = one_instance_gen(n_jobs,n_machines,cloud_features)
        print(times,adj,feat)
        print("-"*40)
