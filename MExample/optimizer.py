from env import *
import numpy as np
import matplotlib.pyplot as plt
import random
np.random.seed(0)
demand_hist_list = []
for k in range(4):
    demand_hist = []
    for i in range(52):
        for j in range(4):
            random_demand = np.random.normal(3, 1.5)
            if random_demand < 0:
                random_demand = 0
            random_demand = np.round(random_demand)
            demand_hist.append(random_demand)
        random_demand = np.random.normal(6, 1)
        if random_demand < 0:
            random_demand = 0
        random_demand = np.round(random_demand)
        demand_hist.append(random_demand)
        for j in range(2):
            random_demand = np.random.normal(12, 2)
            if random_demand < 0:
                random_demand = 0
            random_demand = np.round(random_demand)
            demand_hist.append(random_demand)
    demand_hist_list.append(demand_hist)


def MultiEchelonInvOpt_sS(s_DC,S_DC,s_r1,S_r1,s_r2,S_r2):
    if s_DC > S_DC-1 or s_r1 > S_r1-1 or s_r2 > S_r2-1:
        return -1e8
    else:
        n_retailers = 4
        n_DCs = 2
        retailers = []
        for i in range(n_retailers):
            retailers.append(Retailer(demand_hist_list[i]))
        DCs = []
        for i in range(n_DCs):
            DCs.append(DistributionCenter()) 
        n_period = len(demand_hist_list[0])
        variable_order_cost = 10
        current_period = 1
        total_reward = 0
        while current_period <= n_period:
            action = []
            for DC in DCs:
                if DC.inv_pos <= s_DC:
                    action.append(np.round(min(DC.order_quantity_limit,S_DC-DC.inv_pos)))
                else:
                    action.append(0)
            for i in range(len(retailers)):
                if i%2 == 0:
                    if retailers[i].inv_pos <= s_r1:
                        action.append(np.round(min(retailers[i].order_quantity_limit,S_r1-retailers[i].inv_pos)))
                    else:
                        action.append(0)
                else:
                    if retailers[i].inv_pos <= s_r2:
                        action.append(np.round(min(retailers[i].order_quantity_limit,S_r2-retailers[i].inv_pos)))
                    else:
                        action.append(0)
            y_list = []
            for i in range(n_DCs):
                y = 1 if action[i] > 0 else 0    
                y_list.append(y)
            for DC,order_quantity in zip(DCs,action[:n_DCs]):
                DC.place_order(order_quantity,current_period)
            sum_holding_cost_DC = 0
            for i in range(n_DCs):
                holding_cost_total = DCs[i].order_arrival(retailers[i*2:i*2+2],current_period)
                sum_holding_cost_DC += holding_cost_total
                DCs[i].satisfy_demand(retailers[i*2:i*2+2],action[i*2+2:i*2+4],current_period)
            sum_n_orders = 0
            sum_holding_cost_retailer = 0
            sum_revenue = 0
            for retailer,demand in zip(retailers,demand_hist_list):
                n_orders, holding_cost_total = retailer.order_arrival(current_period)
                sum_n_orders += n_orders
                sum_holding_cost_retailer += holding_cost_total
                revenue = retailer.satisfy_demand(demand[current_period-1])
                sum_revenue += revenue    
            reward = sum_revenue - sum_holding_cost_retailer - sum_holding_cost_DC - sum_n_orders*retailers[0].fixed_order_cost - \
                     np.sum(y_list)*DCs[0].fixed_order_cost - np.sum(action[:n_DCs])*variable_order_cost

            current_period += 1
            total_reward += reward
        return total_reward
    


from bayes_opt import BayesianOptimization
pbounds = {'s_DC': (0,210), 'S_DC': (0, 210), 's_r1': (0, 90), 'S_r1': (0, 90), 's_r2': (0, 90), 'S_r2': (0, 90)}
optimizer = BayesianOptimization(
    f=MultiEchelonInvOpt_sS,
    pbounds=pbounds,
    random_state=0,
)
optimizer.maximize(
    init_points = 100,
    n_iter=1000
)
print(optimizer.max)
