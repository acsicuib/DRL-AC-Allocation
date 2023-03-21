import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from instance_generator import one_instance_gen

from parameters import configs
from GAmodels.placement_GA import MyMutation,MySampling,PlacementProblem,BinaryCrossover
from pymoo.config import Config
from pymoo.indicators.hv import HV
from pymoo.indicators.gd_plus import GDPlus
from pymoo.indicators.gd import GD
from pymoo.indicators.igd_plus import IGDPlus

import time
import warnings
warnings.filterwarnings("ignore")

Config.warnings['not_compiled'] = False
np.random.seed(configs.np_seed_train)

algorithm = NSGA2(
    pop_size=100,
    sampling=MySampling(),
    crossover=BinaryCrossover(prob_mutation=.05), #TODO remove prob_mutation !!BUG?
    mutation=MyMutation(prob=.0), #TODO fix prob
    eliminate_duplicates=True
)

# termination = get_termination("n_gen", 50)
termination = get_termination("n_gen", configs.n_gen)

exec_time,adj,featHW = one_instance_gen(n_jobs=configs.n_jobs, 
                                        n_devices=configs.n_devices,
                                        cloud_features=configs.cloud_features, 
                                        dependency_degree=configs.DAG_rand_dependencies_factor
                                        )

problem = PlacementProblem(n_var=(configs.n_devices+1)*configs.n_tasks,
                           n_objectives=2,
                           time=exec_time,
                           adj=adj,
                           featHW=featHW,
                           n_devices=configs.n_devices,
                           n_tasks=configs.n_tasks) #2 funciones objetivos
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
# print("Pareto function values: %s" % (res.F)) # Pareto
print("Pareto size:: %i" % (len(res.F))) # Pareto

sols = res.opt.get("X")
alloc_sols = [sol.reshape(configs.n_devices+1,configs.n_tasks) for sol in sols]
opsID = [np.argmax(alloc,axis=0) for alloc in alloc_sols]

# print(opsID)

# print(problem.featHW)


ref_point = np.array([200., 180.])
ind = HV(ref_point=ref_point)
print("HV", ind(res.F)) 

print("Perf time ",(end-start))

import matplotlib.pyplot as plt
plt.scatter(res.F[:, 0], res.F[:, 1], s=30, facecolors='none', edgecolors='blue')
plt.title("Objective Space")
plt.xlabel("Time")
plt.ylabel("Cost")
plt.show()

