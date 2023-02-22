import itertools
import numpy as np

action_lists = [[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],[0, 5, 10, 15, 20],[0, 5, 10, 15, 20]]
action_map = [x for x in itertools.product(*action_lists)]

class Retailer():
    def __init__(self, demand_records):
        self.inv_level = 25
        self.inv_pos = 25
        self.order_quantity_limit = 20
        self.holding_cost = 3
        self.lead_time = 2
        self.order_arrival_list = []
        self.backorder_quantity = 0
        self.capacity = 50
        self.demand_list = demand_records
        self.unit_price = 30
        self.fixed_order_cost = 50
    
    def reset(self):
        self.inv_level = 25
        self.inv_pos = 25
        self.order_arrival_list = []
        self.backorder_quantity = 0
        
    def order_arrival(self, current_period):
        n_orders = 0
        if len(self.order_arrival_list) > 0:
            index_list = []
            for j in range(len(self.order_arrival_list)):
                if current_period == self.order_arrival_list[j][0]:
                    self.inv_level = min(self.capacity, self.inv_level + self.order_arrival_list[j][1])
                    n_orders += 1
                    index_list.append(j)
            self.order_arrival_list =  [e for i, e in enumerate(self.order_arrival_list) if i not in index_list]
        holding_cost_total = self.inv_level*self.holding_cost
        return n_orders, holding_cost_total
    
    def satisfy_demand(self, demand):
        units_sold = min(demand, self.inv_level)
        self.inv_level = max(0,self.inv_level-demand)
        self.inv_pos = self.inv_level
        if len(self.order_arrival_list) > 0:
            for j in range(len(self.order_arrival_list)):
                self.inv_pos += self.order_arrival_list[j][1]
        revenue = units_sold*self.unit_price
        return revenue

class DistributionCenter():
    def __init__(self):
        self.inv_level = 100
        self.inv_pos = 100
        self.order_quantity_limit = 100
        self.holding_cost = 1
        self.lead_time = 5
        self.order_arrival_list = []
        self.capacity = 200
        self.fixed_order_cost = 75
    
    def reset(self):
        self.inv_level = 100
        self.inv_pos = 100
        self.order_arrival_list = []
        
    def place_order(self, order_quantity, current_period):
        if order_quantity > 0:
            self.order_arrival_list.append([current_period+self.lead_time, order_quantity])
            
    def order_arrival(self, retailers, current_period):
        if len(self.order_arrival_list) > 0:
            if current_period == self.order_arrival_list[0][0]:
                self.inv_level = min(self.capacity, self.inv_level+self.order_arrival_list[0][1])
                self.order_arrival_list.pop(0)
        holding_cost_total = self.inv_level*self.holding_cost
        return holding_cost_total
        
    def satisfy_demand(self, retailers, actions, current_period):
        quantity_satisfied = [0,0]
        total_backorder = np.sum([retailer.backorder_quantity for retailer in retailers])
        if total_backorder > 0:
            if self.inv_level <= retailers[0].backorder_quantity:
                retailers[0].backorder_quantity -= self.inv_level
                quantity_satisfied[0] += self.inv_level
                self.inv_level = 0
            if self.inv_level > retailers[0].backorder_quantity and self.inv_level <= total_backorder:
                if retailers[0].backorder_quantity == 0:
                    retailers[1].backorder_quantity -= self.inv_level
                    quantity_satisfied[1] += self.inv_level
                else:
                    quantity_left = self.inv_level - retailers[0].backorder_quantity
                    quantity_satisfied[0] += retailers[0].backorder_quantity
                    retailers[0].backorder_quantity = 0
                    quantity_satisfied[1] += quantity_left
                    retailers[1].backorder_quantity -= quantity_left
                self.inv_level = 0
            if self.inv_level > total_backorder:
                if retailers[0].backorder_quantity == 0 and retailers[1].backorder_quantity != 0:
                    quantity_satisfied[1] += retailers[1].backorder_quantity
                    retailers[1].backorder_quantity = 0
                if retailers[0].backorder_quantity != 0 and retailers[1].backorder_quantity == 0:
                    quantity_satisfied[0] += retailers[0].backorder_quantity
                    retailers[0].backorder_quantity = 0
                if retailers[0].backorder_quantity != 0 and retailers[1].backorder_quantity != 0:
                    quantity_satisfied[0] += retailers[0].backorder_quantity
                    quantity_satisfied[1] += retailers[1].backorder_quantity
                    retailers[0].backorder_quantity = 0
                    retailers[1].backorder_quantity = 0
                self.inv_level -= total_backorder
                        
        if self.inv_level > 0:
            if self.inv_level <= actions[0]:
                quantity_satisfied[0] += self.inv_level
                retailers[0].backorder_quantity += actions[0] - self.inv_level
                self.inv_level = 0    
            if self.inv_level > actions[0] and self.inv_level <= np.sum(actions):
                if actions[0] == 0:
                    quantity_satisfied[1] += self.inv_level
                    retailers[1].backorder_quantity += actions[1] - self.inv_level
                else:
                    inv_left = self.inv_level-actions[0]
                    quantity_satisfied[0] += actions[0]
                    quantity_satisfied[1] += inv_left
                    retailers[1].backorder_quantity += actions[1] - inv_left
                self.inv_level = 0
            if self.inv_level > np.sum(actions): 
                if actions[0] == 0 and actions[1] != 0:
                    quantity_satisfied[1] += actions[1]
                if actions[0] != 0 and actions[1] == 0:
                    quantity_satisfied[0] += actions[0]
                if actions[0] != 0 and actions[1] != 0:    
                    quantity_satisfied[0] += actions[0]
                    quantity_satisfied[1] += actions[1]
                self.inv_level -= np.sum(actions)   
        else:
            retailers[0].backorder_quantity += actions[0]
            retailers[1].backorder_quantity += actions[1]  
        
        for i in range(len(retailers)):
            quantity_left = quantity_satisfied[i]
            while quantity_left > 0:
                if quantity_left > retailers[i].order_quantity_limit:
                    retailers[i].order_arrival_list.append([current_period+retailers[i].lead_time, retailers[i].order_quantity_limit])
                    quantity_left -= retailers[i].order_quantity_limit
                else:
                    retailers[i].order_arrival_list.append([current_period+retailers[i].lead_time, quantity_left])
                    quantity_left = 0
                         
        self.inv_pos = self.inv_level
        if len(self.order_arrival_list) > 0:
            for j in range(len(self.order_arrival_list)):
                self.inv_pos += self.order_arrival_list[j][1]
        for retailer in retailers:
            self.inv_pos -= retailer.backorder_quantity


class MultiEchelonInvOptEnv():
    def __init__(self, demand_records):
        self.n_retailers = 2
        self.n_DCs = 1
        self.retailers = []
        for i in range(self.n_retailers):
            self.retailers.append(Retailer(demand_records[i]))
        self.DCs = []
        for i in range(self.n_DCs):
            self.DCs.append(DistributionCenter()) 
        self.n_period = len(demand_records[0])
        self.current_period = 1
        self.day_of_week = 0
        self.state = np.array([DC.inv_pos for DC in self.DCs] + [retailer.inv_pos for retailer in self.retailers] + \
                              self.convert_day_of_week(self.day_of_week))
        self.variable_order_cost = 10
        self.demand_records = demand_records
        
    def reset(self):
        for retailer in self.retailers:
            retailer.reset()
        for DC in self.DCs:
            DC.reset()
        self.current_period = 1
        self.day_of_week = 0 
        self.state = np.array([DC.inv_pos for DC in self.DCs] + [retailer.inv_pos for retailer in self.retailers] + \
                              self.convert_day_of_week(self.day_of_week))
        return self.state
    
    def step(self, action):
        action_modified = action_map[action]
        y_list = []
        for i in range(self.n_DCs):
            y = 1 if action_modified[i] > 0 else 0    
            y_list.append(y)
        for DC,order_quantity in zip(self.DCs,action_modified[:self.n_DCs]):
            DC.place_order(order_quantity,self.current_period)
        sum_holding_cost_DC = 0
        for i in range(self.n_DCs):
            holding_cost_total = self.DCs[i].order_arrival(self.retailers,self.current_period)
            sum_holding_cost_DC += holding_cost_total
            self.DCs[i].satisfy_demand(self.retailers,action_modified[i*2+1:i*2+3],self.current_period)
        sum_n_orders = 0
        sum_holding_cost_retailer = 0
        sum_revenue = 0
        for retailer,demand in zip(self.retailers,self.demand_records):
            n_orders, holding_cost_total = retailer.order_arrival(self.current_period)
            sum_n_orders += n_orders
            sum_holding_cost_retailer += holding_cost_total
            revenue = retailer.satisfy_demand(demand[self.current_period-1])
            sum_revenue += revenue    
        reward = sum_revenue - sum_holding_cost_retailer - sum_holding_cost_DC - sum_n_orders*self.retailers[0].fixed_order_cost - \
                 np.sum(y_list)*self.DCs[0].fixed_order_cost - np.sum(action_modified[:self.n_DCs])*self.variable_order_cost

        self.day_of_week = (self.day_of_week+1)%7
        self.state = np.array([DC.inv_pos for DC in self.DCs] + [retailer.inv_pos for retailer in self.retailers] + \
                              self.convert_day_of_week(self.day_of_week))
        self.current_period += 1
        if self.current_period > self.n_period:
            terminate = True
        else: 
            terminate = False
        return self.state, reward, terminate
    
    def convert_day_of_week(self,d):
        if d == 0:
            return [0, 0, 0, 0, 0, 0]
        if d == 1:
            return [1, 0, 0, 0, 0, 0] 
        if d == 2:
            return [0, 1, 0, 0, 0, 0] 
        if d == 3:
            return [0, 0, 1, 0, 0, 0] 
        if d == 4:
            return [0, 0, 0, 1, 0, 0] 
        if d == 5:
            return [0, 0, 0, 0, 1, 0] 
        if d == 6:
            return [0, 0, 0, 0, 0, 1] 