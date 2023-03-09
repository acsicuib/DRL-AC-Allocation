from models.mlp import MLPActor
from models.mlp import MLPCritic
import torch
import torch.nn as nn
from torch.distributions import Categorical
from models.graphcnn import GraphCNN
from parameters import configs
import torch.nn.functional as F
import sys

class ActorCritic(nn.Module):

    def __init__(self,state_dim,device):
        # super().__init__()
        super(ActorCritic, self).__init__()
        self.feature_extract = GraphCNN(num_layers=configs.num_layers,
                                        num_mlp_layers=configs.num_mlp_layers_feature_extract,
                                        input_dim=configs.input_dim,
                                        hidden_dim=configs.hidden_dim,
                                        neighbor_pooling_type=configs.neighbor_pooling_type,
                                        device=device).to(device)
        
        self.actor  = MLPActor(num_layers=configs.num_mlp_layers_actor,  input_dim=configs.hidden_dim*2, hidden_dim=configs.hidden_dim_actor,  output_dim=1).to(device)
        self.critic = MLPCritic(configs.num_mlp_layers_critic,configs.hidden_dim, configs.hidden_dim_critic, 1).to(device) #NOTE alert! 
        
        self.actorPL  = MLPActor(num_layers=configs.num_mlp_layers_actor,  input_dim=configs.input_dim_device, hidden_dim=configs.hidden_dim_actor,  output_dim=1).to(device)
        self.criticPL = MLPCritic(configs.num_mlp_layers_critic, input_dim=configs.input_dim_device, hidden_dim=configs.hidden_dim_critic,output_dim=1).to(device) #NOTE alert! 

    
    def forward(self, state_ft,state_fm, candidate, mask, adj, graph_pool):
        fts_x = state_ft
        fts_hw = state_fm
   
        # I. Pooling nodes for scheduler part
        h_pooled, h_nodes = self.feature_extract(x=fts_x,
                                                 graph_pool=graph_pool,
                                                 adj=adj)

        dummy = candidate.unsqueeze(-1).expand(-1, configs.n_jobs, h_nodes.size(-1))

        candidate_feature = torch.gather(input=h_nodes.reshape(dummy.size(0), -1, dummy.size(-1)), 
                                         dim =1,
                                        index= dummy)
        
        h_pooled_repeated = h_pooled.unsqueeze(1).expand_as(candidate_feature)

        ## II. Scheduler part
        concateFea = torch.cat((candidate_feature, h_pooled_repeated), dim=-1)
        candidate_scores = self.actor(concateFea)
        mask_reshape = mask.reshape(candidate_scores.size())
        candidate_scores[mask_reshape] = float('-inf')
        pi = F.softmax(candidate_scores, dim=1)
        dist = Categorical(pi.squeeze())
        task_ix_sample = dist.sample()
        dist_logprob = dist.log_prob(task_ix_sample)
        v = self.critic(h_pooled) #TODO CRITIC?

        ## III. Placement part
        fthw_c =  fts_hw.reshape(candidate.size(0), configs.n_devices+1, 2) #TODO number of hw features
        elem = torch.full(size=(candidate.size(0), configs.n_devices+1, configs.n_tasks*(configs.r_n_feat-1)),fill_value=0,dtype=torch.float32)
        # elem = torch.full(size=(1, fts_hw.shape[1],configs.n_tasks*(configs.r_n_feat-1)),fill_value=0,dtype=torch.float32)
        
        for e,task in enumerate(fts_x):
            lm = (e//(configs.n_tasks))
            ix_device = int(task[-1])
            task_pos = (e*2)%((configs.n_tasks)*2)
            elem[lm][ix_device][task_pos:task_pos+1]=task[:1] #TODO HW features 


        concateHWFea = torch.cat((fthw_c, elem), dim=-1)
       
        device_scores = self.actorPL(concateHWFea)
        mhi = F.softmax(device_scores, dim=1)
        distMH = Categorical(mhi.squeeze())
        device_ID = distMH.sample()
        distMH_logprob = distMH.log_prob(device_ID)
        vm = self.criticPL(concateHWFea).squeeze(2) #TODO CRITIC?
        vm = torch.min(vm,1,).values #TODO Alert -> Reduciendo 5 accioens a 1 con la peor probabilidad. Note IV

        return candidate.squeeze()[task_ix_sample], task_ix_sample, pi, v, dist_logprob.detach(), device_ID, mhi, vm, distMH_logprob.detach()
        


    def evaluate(self, states, actions, candidates, masks):
        # print("EVAL")
        # print("\t state ",states.shape)
        # print("\t action ",actions)
    
        action_probs = self.actor(states)
        # print("\t action_probs ",action_probs.shape)
        action_probs = F.softmax(action_probs, dim=0) #TODO V2 move to MLP
        softmax_dist = Categorical(action_probs)
        # print("\t Dist", softmax_dist)
        action_logprobs = softmax_dist.log_prob(actions).reshape(-1)
        # print("\t action_logprob", action_logprobs)

        dist_entropy = softmax_dist.entropy().mean()
        state_values = self.critic(states)
        
        return action_logprobs, state_values, dist_entropy

