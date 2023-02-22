import torch
import configs
device = torch.device(configs.device)


state = torch.Tensor([[3.0900e-02, 0.0000e+00, 4.0000e+00],[4.0000e-04, 0.0000e+00, 2.0000e+00],[6.0000e-04, 0.0000e+00, 4.0000e+00],[8.0000e-04, 0.0000e+00, 4.0000e+00],[3.0000e-04, 0.0000e+00, 1.0000e+00],[1.8000e-03, 0.0000e+00, 4.0000e+00],[3.1800e-02, 0.0000e+00, 4.0000e+00],[1.6000e-03, 0.0000e+00, 4.0000e+00],[1.0000e-03, 0.0000e+00, 4.0000e+00]])

fts_hw = torch.Tensor([0.4000, 0.1667, 0.2000, 0.0333, 0.6000, 0.3333, 0.6000, 0.0333, 1.0000,1.0000])

fts_mh = fts_hw.reshape(5,2).unsqueeze(0)
elem = torch.full(size=(1, 5,configs.n_tasks*(configs.r_n_feat-1)),fill_value=0,dtype=torch.float32,device=device)

print(state)

for e,t in enumerate(state):
    machine = int(t[-1])
    task_pos = e*2
    elem[0][machine][task_pos:task_pos+1]=t[0:1]

print(elem)    