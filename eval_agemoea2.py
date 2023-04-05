import numpy as np
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.algorithms.moo.age2 import AGEMOEA2
import sys
from parameters import configs
# from GAmodel.placement_MOEAD import MyMutation,MySampling,PlacementProblemMOEAD,BinaryCrossover
from GAmodel.placement_GA import MyMutation,MySampling,PlacementProblem,BinaryCrossover
import pickle
from datetime import datetime

from pymoo.config import Config
Config.warnings['not_compiled'] = False

def main():
    np.random.seed(configs.np_seed_train)
    
    configs.name ="E1000v2"
    configs.n_devices = 999
    configs.n_jobs = 9
    configs.n_tasks = 81
    configs.n_gen = 100

    path_dt = 'datasets/dt_TEST_%s_%i_%i.npz'%(configs.name,configs.n_jobs,configs.n_devices)
    
    
    # path_dt = 'datasets/dt_TEST_E999_9_9_999.npz'
    
    dataset = np.load(path_dt)
    dataset = [dataset[key] for key in dataset]
    data = []
    for sample in range(len(dataset[0])):
        data.append((dataset[0][sample],
                     dataset[1][sample],
                     dataset[2][sample],
                     ))
    
    # ref_dirs = get_reference_directions("uniform", 2, n_partitions=15)
    # print(ref_dirs)
   
    # algorithm = MOEAD(
    #     ref_dirs,
    #     n_neighbors=50,
    #     prob_neighbor_mating=0.7,
    #     sampling=MySampling(),
    #     crossover=BinaryCrossover(prob_mutation=.15), #TODO remove prob_mutation !!BUG?
    #     mutation=MyMutation(prob=.0), #TODO fix prob
    # )
    
    algorithm = AGEMOEA2(
        pop_size=200,
        sampling=MySampling(),
        crossover=BinaryCrossover(prob_mutation=.15), #TODO remove prob_mutation !!BUG?
        mutation=MyMutation(prob=.0), #TODO fix prob
        eliminate_duplicates=True
    )

    

    termination = get_termination("n_gen", configs.n_gen)

    for i, sample  in enumerate(data):
        if i == 1: break
        print("Running episode: %i"%(i))
        times, adj, feat = sample
        problem = PlacementProblem(n_var=(configs.n_devices+1)*configs.n_tasks,
                                   n_objectives=2, #TODO fix 2 funciones objetivos
                                   time=times,
                                   adj=adj,
                                   featHW=feat,
                                   n_devices=configs.n_devices,
                                   n_tasks=configs.n_tasks
                                   ) 

        sttime = datetime.now().replace(microsecond=0)
        # res = minimize(problem,
        #             algorithm,
        #             termination,
        #             seed=1,
        #             save_history=True,
        #             verbose=False)
        res = minimize(problem,
                    algorithm,
                    termination,
                    seed=1,
                    save_history=True,
                    verbose=False)
 

        ettime = datetime.now().replace(microsecond=0)
        
        print(res.X.shape)
        try:
            log_pf = []
            for ix,pf in enumerate(res.X):
                if res.X.shape[0]==configs.n_tasks*(configs.n_devices+1):
                    # print("A")
                    solution = problem.myevaluate(res.X)
                else:
                    # print("C")
                    solution = problem.myevaluate(res.X[ix])

                log_pf.append([i,solution[0],solution[1],solution[2],(ettime-sttime)])

                with open('logs/log_AGEMOEA2_pf'+ str(configs.name) + "_" + str(configs.n_jobs) + '_' + str(configs.n_devices)+'_%i_w%s.pkl'%(i,codeW), 'wb') as f:
                        pickle.dump(log_pf, f)
        except Exception as e: 
            print("\tProblem with CASE %i"%i)
            print(e)
            print("*"*10)

        print("\tEpisode %i \t\t time:%s"%(i,str((ettime-sttime))))

        

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
