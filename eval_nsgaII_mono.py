import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize

from parameters import configs
from GAmodel.placement_NSGA import MyMutation,MySampling,MonoPlacementProblem,BinaryCrossover
import pickle
from datetime import datetime

import sys
from pymoo.config import Config
Config.warnings['not_compiled'] = False


def main():
    np.random.seed(configs.np_seed_train)
    
    # ## DEBUG
    # configs.name ="E999_9"
    # configs.n_devices = 999
    # configs.n_jobs = 9
    # configs.n_tasks = 81
    # configs.n_gen = 4
    # ###



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
        pop_size=100,
        sampling=MySampling(),
        crossover=BinaryCrossover(prob_mutation=.15), #TODO remove prob_mutation !!BUG?
        mutation=MyMutation(prob=.0), #TODO fix prob
        eliminate_duplicates=True,
        save_history = True
    )

    codeW = str(int(configs.rewardWeightTime*100))+str(int(configs.rewardWeightCost*100))
    termination = get_termination("n_gen", configs.n_gen)

    for i, sample  in enumerate(data):
    # for i, sample  in enumerate(data):
        if i == 1: break
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
                                   wCost=configs.rewardWeightCost,
                                   norm_time = configs.norm_time,
                                   norm_cost = configs.norm_cost) 

        sttime = datetime.now().replace(microsecond=0)
        res = minimize(problem,
                    algorithm,
                    termination,
                    seed=1,
                    save_history=True,
                    verbose=False)
        
        ettime = datetime.now().replace(microsecond=0)

        # print(res.X)
        # print("_")
        # print(res.X.shape)
        


        # print("History")
        # n_evals = np.array([e.evaluator.n_eval for e in res.history])
        # opt = np.array([e.opt[0].F for e in res.history])
        convergence = [res.history[i].result().f for i in range(len(res.history))]
        exec_time = [res.history[i].result().exec_time for i in range(len(res.history))]
        ct = zip(convergence,exec_time)
        # print(convergence)
        # print(n_evals)
        # print(opt)
        # print(len(convergence))
        with open('logs/log_ga_mono_convergence'+ str(configs.name) + "_" + str(configs.n_jobs) + '_' + str(configs.n_devices)+'_%i_w%s.pkl'%(i,codeW), 'wb') as f:
                    pickle.dump(ct, f)
        # print(convergence)

        # sys.exit()
        
        try:
            log_pf = []
            for ix,pf in enumerate(res.X):
                if res.X.shape[0]==configs.n_tasks*(configs.n_devices+1):
                    # print("A")
                    solution = problem.myevaluate(res.X)
                elif res.X.shape[0]==40500:
                    solution = problem.myevaluate(res.X)
                else:
                    # print("C")
                    solution = problem.myevaluate(res.X[ix])

                log_pf.append([i,solution[0],solution[1],solution[2],(ettime-sttime)])

            with open('logs/log_ga_mono_'+ str(configs.name) + "_" + str(configs.n_jobs) + '_' + str(configs.n_devices)+'_%i_w%s.pkl'%(i,codeW), 'wb') as f:
                    pickle.dump(log_pf, f)
        except Exception as e: 
            print("\tProblem with CASE %i"%i)
            print(e)
            print("*"*10)
                
        print('\t ending episode {}\t Len PF: {}\t'.format(i , len(res.F)))
        print("\t time: ",(ettime-sttime))
        print("\t NEXT")
        # break
        

if __name__ == '__main__':
    print("NSGAII-MONO-strategy")
    start_time = datetime.now().replace(microsecond=0)
    print("Started training: ", start_time)
    print("="*30)
    main()
    end_time = datetime.now().replace(microsecond=0)
    print("Finish training: ", end_time)
    print("Total time: ",(end_time-start_time))
    print("Done.")
