import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize

from parameters import configs
from GAmodel.placement_GA import MyMutation,MySampling,PlacementProblem,BinaryCrossover
from pymoo.config import Config

Config.warnings['not_compiled'] = False
np.random.seed(configs.np_seed_train)

algorithm = NSGA2(
    pop_size=100,
    sampling=MySampling(),
    crossover=BinaryCrossover(prob_mutation=.05), #TODO remove prob_mutation !!BUG?
    mutation=MyMutation(prob=.0), #TODO fix prob
    eliminate_duplicates=True
)

termination = get_termination("n_gen", 50)


problem = PlacementProblem(n_var=(configs.n_devices+1)*configs.n_tasks,n_objectives=2) #2 funciones objetivos

res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               save_history=True,
               verbose=True)

print("Pareto function values: %s" % (res.F)) # Pareto

sols = res.opt.get("X")
alloc_sols = [sol.reshape(configs.n_devices+1,configs.n_tasks) for sol in sols]
opsID = [np.argmax(alloc,axis=0) for alloc in alloc_sols]

print(opsID)

print(problem.featHW)

import matplotlib.pyplot as plt
plt.scatter(res.F[:, 0], res.F[:, 1], s=30, facecolors='none', edgecolors='blue')
plt.title("Objective Space")
plt.xlabel("Time")
plt.ylabel("Cost")
plt.show()