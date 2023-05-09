import numpy as np
import pickle
from datetime import datetime
from parameters import configs
import networkx as nx

def main():
    codeW = str(int(configs.rewardWeightTime*10))+str(int(configs.rewardWeightCost*10))
    print(codeW)
    path_dt = 'datasets/dt_TEST_%s_%i_%i.npz'%(configs.name,configs.n_jobs,configs.n_devices)
    dataset = np.load(path_dt)
    dataset = [dataset[key] for key in dataset]
    data = []
    for sample in range(len(dataset[0])):
        data.append((dataset[0][sample],
                     dataset[1][sample],
                     dataset[2][sample],
                     ))
    G = None
    for i, sample  in enumerate(data):
        times,adj,feat = sample
        # print(times)

        adj = np.array(adj)
        print(adj.shape)
        np.fill_diagonal(adj, 0)
        print(adj)
        G = nx.from_numpy_matrix(adj)
        
        
        break

    print(G.nodes)
    lat,lng = {},{}
    
    for n in G.nodes:
        lat[n]=n%configs.n_jobs
        lng[n]=n//configs.n_jobs
    
    nx.set_node_attributes(G, lat, "lat")
    nx.set_node_attributes(G, lng, "lng")
    
    nx.write_gexf(G, "results/adj_TEST_E1000_app0.gexf")

if __name__ == '__main__':
    main()