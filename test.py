import torch
import configs
import numpy as np
device = torch.device(configs.device)


state = torch.Tensor([[3.0900e-02, 0.0000e+00, 4.0000e+00],[4.0000e-04, 0.0000e+00, 2.0000e+00],[6.0000e-04, 0.0000e+00, 4.0000e+00],[8.0000e-04, 0.0000e+00, 4.0000e+00],[3.0000e-04, 0.0000e+00, 1.0000e+00],[1.8000e-03, 0.0000e+00, 4.0000e+00],[3.1800e-02, 0.0000e+00, 4.0000e+00],[1.6000e-03, 0.0000e+00, 4.0000e+00],[1.0000e-03, 0.0000e+00, 4.0000e+00]])

fts_hw = torch.Tensor([0.4000, 0.1667, 0.2000, 0.0333, 0.6000, 0.3333, 0.6000, 0.0333, 1.0000,1.0000])

fts_mh = fts_hw.reshape(5,2).unsqueeze(0)
elem = torch.full(size=(1, 5,configs.n_tasks*(configs.r_n_feat-1)),fill_value=0,dtype=torch.float32,device=device)

i = np.array([2,8,0, 8 ,7 ,7 ,3 ,2, 3])
fw = np.array([[ 6.,  3.,  5.,  4.,],
 [ 2.,  3.,  1.,  4.,],
 [ 4.,  3.,  5.,  4.,],
 [ 4.,  3.,  5.,  4.,],
 [ 2.,  3.,  5.,  4.,],
 [ 4.,  1.,  5.,  4.,],
 [ 2.,  3.,  5.,  4.,],
 [ 6.,  3., 10.,  4.,],
 [ 2.,  3.,  1.,  4.,],
 [10.,  1., 30., 10.,]])

print(fw)
print(fw[i][0])