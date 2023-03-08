import numpy as np
from DAG_app_generator import generate_DAG_application

from parameters import configs 

def one_instance_gen(n_jobs,n_devices,cloud_features,dependency_degree):

    times, adj = generate_DAG_application(n_jobs,configs.task_time_low,configs.task_time_high,degree=dependency_degree)
    n_features = len(configs.feature_labels)
    
    feat_Speed = np.random.choice(configs.cpu_speed_options,n_devices)
    feat_Cost = np.random.choice(configs.cost_options,n_devices)
    feat_Lat = np.random.choice(configs.latency_options,n_devices)
    feat_Load = np.repeat([n_features],n_devices) #TODO HW features
    # feat_LoadPena = np.zeros(n_machines)  
    # feat = np.concatenate((feat_HW,feat_Cost,feat_Lat,feat_Load,feat_LoadPena)).reshape(n_features,n_machines).T
    
    feat = np.concatenate((feat_Speed,feat_Cost,feat_Lat,feat_Load)).reshape(n_features,n_devices).T
    
    # last machine represents the cloud entity
    feat = np.vstack((feat,cloud_features)) #Cloud is always the same
    

    
    # ! torch.float
    feat = feat.astype(np.float32)
    
    return times, adj, feat

if __name__ == '__main__':
    n_jobs = 3
    n_devices = 5 # In total = 5+1, cloud entity
    cloud_features = [20,2,4,0]
    degree = 0.2
    for i in range(10):
        times, adj,feat = one_instance_gen(n_jobs,n_devices,cloud_features,degree)
        print(times,adj,feat)
        print("-"*40)
