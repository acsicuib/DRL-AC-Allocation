import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
import sys
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from instance_generator import one_instance_gen
from environment.utils import getCNTimes,getCNCosts

from parameters import configs
from GAmodel.problem_GA import MyMutation,MySampling,GAPlacementProblem,BinaryCrossover
from pymoo.config import Config
from pymoo.indicators.hv import HV
from pymoo.indicators.gd_plus import GDPlus
from pymoo.indicators.gd import GD
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.algorithms.soo.nonconvex.es import ES
import time
import warnings
warnings.filterwarnings("ignore")

Config.warnings['not_compiled'] = False
np.random.seed(configs.np_seed_train)

pop_size = 200
algorithm = GA(
    pop_size=pop_size,
    eliminate_duplicates=True,
    crossover=BinaryCrossover(prob_mutation=.05), #TODO remove prob_mutation !!BUG?
    mutation=MyMutation(prob=.0), #TODO fix prob
    sampling=MySampling())

n_jobs = 9
n_tasks = n_jobs*n_jobs
devices = 500
n_gen = 10
termination = get_termination("n_gen", n_gen)

times,adj,feat = one_instance_gen(n_jobs=n_jobs, 
                                        n_devices=devices,
                                        cloud_features=configs.cloud_features, 
                                        dependency_degree=configs.DAG_rand_dependencies_factor
                                        )
# print(times.shape)
# print(adj.shape)
# print(feat.shape)
# print(feat)


problem = GAPlacementProblem(n_var=(devices+1)*n_tasks,
                                   n_objectives=1, #TODO fix 2 funciones objetivos
                                   time=times,
                                   adj=adj,
                                   featHW=feat,
                                   n_devices=devices,
                                   n_tasks=n_tasks,
                                   pop_size = pop_size,
                                   wTime=configs.rewardWeightTime, 
                                   wCost=configs.rewardWeightCost,
                                   ratioRule = 100,
                                   norm_time = configs.norm_time,
                                   norm_cost = configs.norm_cost) 


start = time.perf_counter()
print("Running the model")

res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               save_history=True,
               verbose=False,
               return_least_infeasible=True)


end = time.perf_counter()
# print("Sol:", (len(res.X))) 
# print("Sol:", (res.X)) 
sol =  res.X.reshape(devices+1,n_tasks)

f1 = np.sum(getCNTimes(sol,times,feat,adj))
f2 = np.sum(getCNCosts(sol,feat))
print("time  :",f1)
print("cost  :",f2)
# alloc_sols = [sol.reshape(configs.n_devices+1,configs.n_tasks) for sol in sols]
# opsID = [np.argmax(alloc,axis=0) for alloc in alloc_sols]

# print(opsID)

# print(problem.featHW)


# ref_point = np.array([200., 180.])
# ind = HV(ref_point=ref_point)
# print("HV", ind(res.F)) 

print("Perf time ",(end-start))

# import matplotlib.pyplot as plt
# plt.scatter(res.F[:, 0], res.F[:, 1], s=30, facecolors='none', edgecolors='blue')
# plt.title("Objective Space")
# plt.xlabel("Time")
# plt.ylabel("Cost")
# plt.show()

