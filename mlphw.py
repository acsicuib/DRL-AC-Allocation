import torch
import torch.nn as nn
import torch.nn.functional as F
from mlp import MLP
import configs
import numpy as np
# import sys
# sys.path.append("models/")


class MLPHW(nn.Module):
    def __init__(self,
                 num_layers,
                 num_mlp_layers,
                 input_dim,
                 hidden_dim,
                 learn_eps,
                 neighbor_pooling_type,
                 device):
        '''
        num_layers: number of layers in the neural networks (INCLUDING the input layer)
        num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
        input_dim: dimensionality of input features
        hidden_dim: dimensionality of hidden units at ALL layers
        output_dim: number of classes for prediction
        final_dropout: dropout ratio on the final linear layer
        learn_eps: If True, learn epsilon to distinguish center nodes from neighboring nodes. If False, aggregate neighbors and center nodes altogether.
        neighbor_pooling_type: how to aggregate neighbors (mean, average, or max)
        device: which device to use
        '''

        super(MLPHW, self).__init__()

        # self.final_dropout = final_dropout
        self.device = device
        self.num_layers = num_layers
        self.learn_eps = learn_eps

        # List of MLPs
        self.mlps = torch.nn.ModuleList()
        self.neighbor_pooling_type = neighbor_pooling_type
        
        # List of batchnorms applied to the output of MLP (input of the final prediction linear layer)
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers-1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def next_layer(self, h, layer, feat_hw,):

        # pooling neighboring nodes and center nodes altogether
        pooled = torch.mm(feat_hw, h)
        # representation of neighboring and center nodes
        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)

        # non-linearity
        h = F.relu(h)
        return h

    def forward(self,
                x,
                feat_hw
                ):
        h = x

        for layer in range(self.num_layers-1):
            # print("\tLayer ",layer)
            h = self.next_layer(h, layer, feat_hw)

        hw_nodes = h.clone()

        # print("H_nodes:",h_nodes.shape)
        return hw_nodes


if __name__ == '__main__':
    import configs
    from dag_aggregate import g_pool_cal
    device = torch.device(configs.device)

    model = MLPHW(configs.num_layers,
                  configs.num_mlp_layers_feature_hw,
                  configs.input_dim,
                  configs.hidden_dim,
                  configs.learn_eps,
                  configs.neighbor_pooling_type,
                  device)

    print(model)
    state = torch.Tensor([3.0900e-02, 0.0000e+00, 4.0000e+00, 4.0000e-04, 0.0000e+00, 4.0000e+00,
        6.0000e-04, 0.0000e+00, 4.0000e+00, 8.0000e-04, 0.0000e+00, 4.0000e+00,
        3.0000e-04, 0.0000e+00, 4.0000e+00, 1.8000e-03, 0.0000e+00, 4.0000e+00,
        3.1800e-02, 0.0000e+00, 4.0000e+00, 1.6000e-03, 0.0000e+00, 4.0000e+00,
        1.0000e-03, 0.0000e+00, 4.0000e+00]).reshape(9,3)
    
    ft_hw = torch.Tensor([0.4000, 0.1667, 0.2000, 0.0333, 0.6000, 0.3333, 0.6000, 0.0333, 1.0000,
        1.0000]).reshape(5,2)

    print(state.shape)       
    print(ft_hw.shape)       
    # hwnodes = model.forward(state,ft_hw)
    # print(hwnodes)
    # print(hwnodes.shape)
    x = torch.Tensor([4]*4).reshape(4,1)
    y = torch.Tensor([0,0,0,1]).reshape(1,4)
    print(x)
    print(y)
    ttt = torch.mm(x,y)
    print(ttt)

    graph_pool = g_pool_cal("average",torch.Size([1, 9, 9]),9,device) #9 tasks


    print(graph_pool)

    pooled_hw = torch.sparse.mm(graph_pool, ft_hw)
    print(pooled_hw)
