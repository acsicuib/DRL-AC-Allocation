import numpy as np
from environment.DAG_app_generator import generate_DAG_application
from parameters import configs 
import sys
def one_instance_gen(n_jobs,n_devices,cloud_features,dependency_degree):

    times, adj = generate_DAG_application(n_jobs,configs.task_time_low,configs.task_time_high,degree=dependency_degree)
    n_features = len(configs.feature_labels)
    
    feat_Speed = np.random.choice(configs.cpu_speed_options,n_devices)
    feat_Load = np.repeat([n_features],n_devices) #TODO HW features

    
    ixCs = np.random.randint(0,len(configs.cost_options),n_devices)
    ixTs = abs(ixCs-(len(configs.cost_options)-1))

    feat_Cost = np.take(configs.cost_options,ixCs)
    feat_Lat = np.take(configs.latency_options,ixTs)
    
    # feat_LoadPena = np.zeros(n_machines)  
    # feat = np.concatenate((feat_HW,feat_Cost,feat_Lat,feat_Load,feat_LoadPena)).reshape(n_features,n_machines).T
    
    feat = np.concatenate((feat_Speed,feat_Cost,feat_Lat,feat_Load)).reshape(n_features,n_devices).T
    
    # last machine represents the cloud entity
    feat = np.vstack((feat,cloud_features)) #Cloud is always the same
    

    
    # ! torch.float
    feat = feat.astype(np.float32)
    
    return times, adj, feat

if __name__ == '__main__':
    print("Test one_instance_gen function")
    n_jobs = 5
    n_devices = 1000 # In total = 5+1, cloud entity
    # cloud_features = [20,10,4,0]
    cloud_features = configs.cloud_features
    degree = 0.2
    for i in range(10):
        times, adj,feat = one_instance_gen(n_jobs,n_devices,cloud_features,degree)
        print("Times: \n",times)
        print("AdjM: \n",adj)
        # print("Feat: %s\n"%configs.feature_labels,feat)
        # print(times),adj,feat)
        print("-"*40)

        v,c = np.unique(feat[:,1],return_counts=True)
        print(v,"\n",c)
        v,c = np.unique(feat[:,2],return_counts=True)
        print(v,"\n",c)

        break
