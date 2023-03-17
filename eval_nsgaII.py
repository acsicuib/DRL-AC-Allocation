import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize

from parameters import configs
from GAmodel.placement_GA import MyMutation,MySampling,PlacementProblem,BinaryCrossover
import pickle
from datetime import datetime

def main():
    np.random.seed(configs.np_seed_train)
    
    path_dt = 'datasets/dt_%s_%i_%i.npz'%(configs.name,configs.n_jobs,configs.n_devices)
    dataset = np.load(path_dt)
    dataset = [dataset[key] for key in dataset]
    data = []
    for sample in range(len(dataset[0])):
        data.append((dataset[0][sample],
                     dataset[1][sample],
                     dataset[2][sample],
                     ))
    
    algorithm = NSGA2(
        pop_size=100,
        sampling=MySampling(),
        crossover=BinaryCrossover(prob_mutation=.08), #TODO remove prob_mutation !!BUG?
        mutation=MyMutation(prob=.0), #TODO fix prob
        eliminate_duplicates=True
    )

    termination = get_termination("n_gen", 50)

    log_pf = []
    for i, sample  in enumerate(data):
        times, adj, feat = sample
        problem = PlacementProblem(n_var=(configs.n_devices+1)*configs.n_tasks,
                                   n_objectives=2, #2 funciones objetivos
                                   time=times,
                                   adj=adj,
                                   featHW=feat,
                                   n_devices=configs.n_devices,
                                   n_tasks=configs.n_tasks) 

        res = minimize(problem,
                    algorithm,
                    termination,
                    seed=1,
                    save_history=True,
                    verbose=False)

        for pf in res.F:
            log_pf.append([i,pf[0],pf[1],res.exec_time])

        print('\tEpisode {}\t Len PF: {}\t exec time {:.2f}'.format(i + 1, len(res.F), res.exec_time))

    if True:
        with open('logs/log_ga_pf_'+ str(configs.name) + "_" + str(configs.n_jobs) + '_' + str(configs.n_devices)+'.pkl', 'wb') as f:
            pickle.dump(log_pf, f)

if __name__ == '__main__':
    print("NSGAII-strategy test: using default parameters")
    start_time = datetime.now().replace(microsecond=0)
    print("Started training: ", start_time)
    print("="*30)
    main()
    end_time = datetime.now().replace(microsecond=0)
    print("Finish training: ", end_time)
    print("Total time: ",(end_time-start_time))
    print("Done.")
