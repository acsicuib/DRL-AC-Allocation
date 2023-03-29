import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize

from parameters import configs
from GAmodel.placement_GA import MyMutation,MySampling,MonoPlacementProblem,BinaryCrossover
import pickle
from datetime import datetime

from pymoo.config import Config
Config.warnings['not_compiled'] = False

def main():
    np.random.seed(configs.np_seed_train)
    
    ## DEBUG
    # configs.name ="E999_9"
    # configs.n_devices = 999
    # configs.n_jobs = 9
    # configs.n_tasks = 81
    # configs.n_gen = 4


    path_dt = 'datasets/dt_TEST_%s_%i_%i.npz'%(configs.name,configs.n_jobs,configs.n_devices)
    dataset = np.load(path_dt)
    dataset = [dataset[key] for key in dataset]
    data = []
    for sample in range(len(dataset[0])):
        data.append((dataset[0][sample],
                     dataset[1][sample],
                     dataset[2][sample],
                     ))
    
    algorithm = NSGA2(
        pop_size=200,
        sampling=MySampling(),
        crossover=BinaryCrossover(prob_mutation=.15), #TODO remove prob_mutation !!BUG?
        mutation=MyMutation(prob=.0), #TODO fix prob
        eliminate_duplicates=True
    )

    termination = get_termination("n_gen", configs.n_gen)

    for i, sample  in enumerate(data):
        
        print("Running episode: %i"%(i+1))
        times, adj, feat = sample
        problem = MonoPlacementProblem(n_var=(configs.n_devices+1)*configs.n_tasks,
                                   n_objectives=1, #TODO fix 2 funciones objetivos
                                   time=times,
                                   adj=adj,
                                   featHW=feat,
                                   n_devices=configs.n_devices,
                                   n_tasks=configs.n_tasks,
                                   wTime=configs.rewardWeightTime, 
                                   wCost=configs.rewardWeightCost) 

        sttime = datetime.now().replace(microsecond=0)
        res = minimize(problem,
                    algorithm,
                    termination,
                    seed=1,
                    save_history=True,
                    verbose=False)
        
        ettime = datetime.now().replace(microsecond=0)

        print(res.X)
        

        log_pf = []
        for ix,pf in enumerate(res.F):
            solution = problem.myevaluate(res.X[ix])
            log_pf.append([i,pf,solution[0],solution[1],solution[2],(ettime-sttime)])

        with open('logs/log_ga_pf_mono_'+ str(configs.name) + "_" + str(configs.n_jobs) + '_' + str(configs.n_devices)+'_%i.pkl'%i, 'wb') as f:
                pickle.dump(log_pf, f)
                
        print('\tEpisode {}\t Len PF: {}\t'.format(i + 1, len(res.F)))
        # print("\t\t time: ",(ettime-sttime))
        # break
        

if __name__ == '__main__':
    print("NSGAII-MONO-strategy test: using default parameters")
    start_time = datetime.now().replace(microsecond=0)
    print("Started training: ", start_time)
    print("="*30)
    main()
    end_time = datetime.now().replace(microsecond=0)
    print("Finish training: ", end_time)
    print("Total time: ",(end_time-start_time))
    print("Done.")