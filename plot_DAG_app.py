import networkx as nx
from parameters import configs 
from instance_generator import one_instance_gen
import numpy as np
n_jobs = 3
n_devices = 15 # In total = 5+1, cloud entity
# cloud_features = [20,10,4,0]
cloud_features = configs.cloud_features

times, adj,feat = one_instance_gen(n_jobs,n_devices,cloud_features,configs.DAG_rand_dependencies_factor)

np.fill_diagonal(adj, 0)
G = nx.from_numpy_matrix(adj)
print(G.nodes)
lat,lng = {},{}

for n in G.nodes:
    lat[n]=n%configs.n_jobs
    lng[n]=n//configs.n_jobs

nx.set_node_attributes(G, lat, "lat")
nx.set_node_attributes(G, lng, "lng")

nx.write_gexf(G, "results/adj_plot.gexf")
