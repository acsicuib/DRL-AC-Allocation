import numpy as np
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

import sys
from parameters import configs
from GAmodel.placement_MOEAD import MyMutation,MySampling,PlacementProblem,BinaryCrossover
import pickle
from datetime import datetime

from pymoo.config import Config
Config.warnings['not_compiled'] = False

def main():
    np.random.seed(configs.np_seed_train)
    
    path_dt = 'datasets/dt_TEST_%s_%i_%i.npz'%(configs.name,configs.n_jobs,configs.n_devices)
    dataset = np.load(path_dt)
    dataset = [dataset[key] for key in dataset]
    data = []
    for sample in range(len(dataset[0])):
        data.append((dataset[0][sample],
                     dataset[1][sample],
                     dataset[2][sample],
                     ))
    
    ref_dirs = get_reference_directions("uniform", 2, n_partitions=12)
    # print(ref_dirs)
   
    algorithm = MOEAD(
        ref_dirs,
        n_neighbors=15,
        prob_neighbor_mating=0.7,
    )
    

    termination = get_termination("n_gen", configs.n_gen)

    for i, sample  in enumerate(data):
        
        print("Running episode: %i"%(i+1))
        times, adj, feat = sample
        problem = PlacementProblem(n_var=(configs.n_devices+1)*configs.n_tasks,
                                   n_objectives=2, #TODO fix 2 funciones objetivos
                                   time=times,
                                   adj=adj,
                                   featHW=feat,
                                   n_devices=configs.n_devices,
                                   n_tasks=configs.n_tasks) 

        sttime = datetime.now().replace(microsecond=0)
        res = minimize(problem,
                    algorithm,
                    termination,
                    seed=1,
                    save_history=True,
                    verbose=False)
        
        ettime = datetime.now().replace(microsecond=0)
        
        log_pf = []
        for pf in res.F:
            log_pf.append([i,pf[0],pf[1],str((ettime-sttime))])

        with open('logs/log_moead_pf_'+ str(configs.name) + "_" + str(configs.n_jobs) + '_' + str(configs.n_devices)+'_%i.pkl'%i, 'wb') as f:
                pickle.dump(log_pf, f)

        print('\tEpisode {}\t Len PF: {}\t exec time {:.2f}'.format(i + 1, len(res.F)))
        print("\t\t time: ",str((ettime-sttime)))

        

if __name__ == '__main__':
    print("MOEAD-strategy test: using default parameters")
    start_time = datetime.now().replace(microsecond=0)
    print("Started training: ", start_time)
    print("="*30)
    main()
    end_time = datetime.now().replace(microsecond=0)
    print("Finish training: ", end_time)
    print("Total time: ",(end_time-start_time))
    print("Done.")
