
from instance_generator import one_instance_gen
from parameters import configs
import numpy as np
import pickle as pkl

CASE = "VALIDATION"

CASE = "TEST"



with open('datasets/dt_%s_%s_%i_%i.npz'%(CASE,configs.name,configs.n_jobs,configs.n_devices), 'wb') as f:


    np.random.seed(configs.np_seed_dataset)

    times, adj, feat= [],[],[]
    for i in range(configs.len_dataset):
        data = one_instance_gen(n_jobs=configs.n_jobs, 
                                n_devices=configs.n_devices,
                                cloud_features=configs.cloud_features, 
                                dependency_degree=configs.DAG_rand_dependencies_factor)
        times.append(data[0])
        adj.append(data[1])
        feat.append(data[2])
    
    np.savez(f,times=times,adj=adj,feat=feat)
        


