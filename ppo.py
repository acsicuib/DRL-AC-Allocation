import torch
import torch.nn as nn
from torch.distributions import Categorical
import configs
import torch.nn.functional as F
from copy import deepcopy
from env import descomposeAction #V3
from graphcnn import GraphCNN
from doubleActorCritic import ActorCritic
from mb_agg import aggr_obs, g_pool_cal

import sys

device = torch.device(configs.device)

class Memory:
    def __init__(self):
        self.alloc_mb = []
        self.adj_mb = []
        self.featTask = []
        self.featMach = []
        self.state_ft = []
        self.state_fm = []
        self.candidate_mb = []
        self.mask_mb = []
        self.a_mb = []#action_task
        self.am_mb = [] #action_ machine
        self.reward_mb = []
        self.done_mb = []
        self.logprobs = []
        self.logprobs_m = [] 

    def clear_memory(self):
        del self.alloc_mb[:]
        del self.adj_mb[:]
        del self.featTask[:]
        del self.featMach[:]
        del self.candidate_mb[:]
        del self.mask_mb[:]
        del self.a_mb[:]
        del self.am_mb[:]
        del self.reward_mb[:]
        del self.done_mb[:]
        del self.logprobs[:]
        del self.logprobs_m[:] 
        del self.state_ft[:]
        del self.state_fm[:]

class PPO:
    def __init__(self, state_dim):

        self.gamma = configs.gamma
        self.eps_clip = configs.eps_clip
        self.k_epochs = configs.k_epochs
        
        # self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim,device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': configs.lr_agent},
                        {'params': self.policy.critic.parameters(), 'lr': configs.lr_critic}
                    ])

        self.policy_old =  deepcopy(self.policy)

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=configs.decay_step_size,
                                                         gamma=configs.decay_ratio)

        self.MseLoss = nn.MSELoss()
        

    def update(self,memories):
        vloss_coef = configs.vloss_coef
        ploss_coef = configs.ploss_coef
        entloss_coef = configs.entloss_coef
    
        # Monte Carlo estimate of returns
        
        adj_mb_t_all_env = []
        alloc_mb_t_all_env = []
        state_ft_all_env, state_fm_all_env = [], []
        # fea_mb_t_all_env = []
        # fea_mb_m_all_env = []
        candidate_mb_t_all_env = []
        mask_mb_t_all_env = []
        rewards_all_env = []
        a_mb_t_all_env = []# task action
        am_mb_t_all_env = []# machine action
        old_logprobs_mb_t_all_env = []
        old_logprobs_m_mb_t_all_env = []

        # store data for all env
        for i in range(len(memories)):
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(memories[i].reward_mb), reversed(memories[i].done_mb)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
            rewards = torch.tensor(rewards, dtype=torch.float).to(device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            rewards_all_env.append(rewards)

            # process each env data
            adj_mb_t_all_env.append(aggr_obs(torch.stack(memories[i].adj_mb).to(device), configs.n_tasks))
            
            ft_tensor = torch.stack(memories[i].state_ft).to(device)
            ft_tensor = ft_tensor.reshape(-1, ft_tensor.size(-1))
            state_ft_all_env.append(ft_tensor)

            fm_tensor = torch.stack(memories[i].state_fm).to(device)
            fm_tensor = fm_tensor.reshape(-1, fm_tensor.size(-1)).unsqueeze(0)
            state_fm_all_env.append(fm_tensor)
        
           
            alloc_mb_t_all_env.append(torch.stack(memories[i].alloc_mb).to(device)) #TODO comprobar si se puede borrar ent odo el proyecto
            candidate_mb_t_all_env.append(torch.stack(memories[i].candidate_mb).to(device).squeeze())
            mask_mb_t_all_env.append(torch.stack(memories[i].mask_mb).to(device).squeeze())
            a_mb_t_all_env.append(torch.stack(memories[i].a_mb).to(device).squeeze())
            am_mb_t_all_env.append(torch.stack(memories[i].am_mb).to(device).squeeze())
            old_logprobs_mb_t_all_env.append(torch.stack(memories[i].logprobs).to(device).squeeze().detach())
            old_logprobs_m_mb_t_all_env.append(torch.stack(memories[i].logprobs_m).to(device).squeeze().detach())

        
        mb_g_pool = g_pool_cal(configs.graph_pool_type, torch.stack(memories[0].adj_mb).to(device).shape, configs.n_tasks, device)

        # Optimize policy for K epochs:
        for _ in range(self.k_epochs):
            loss_sum = 0
            vloss_sum = 0

            for i in range(len(memories)):
                # old_states = torch.tensor(torch.cat((fea_mb_t_all_env[i][i].reshape(-1),feat_mach_tensor_envs[i].reshape(-1))),dtype=torch.float)
                # old_states = torch.concatenate((fea_mb_t_all_env[i],fea_mb_m_all_env[i]))
                
                # logprobs, state_values, dist_entropy_loss = self.policy(
                #     states=state_mb_t_all_env[i], 
                #     actions=a_mb_t_all_env[i],
                #     candidates=candidate_mb_t_all_env[i],
                #     masks=mask_mb_t_all_env[i]
                #     )
                
                task_action, vals, log_taskprob, machine_action, valsm, log_machprob = self.policy(
                    state_ft=state_ft_all_env[i],
                    state_fm=state_fm_all_env[i],
                    candidate=candidate_mb_t_all_env[i],
                    mask=mask_mb_t_all_env[i],
                    adj=adj_mb_t_all_env[i],
                    graph_pool=mb_g_pool
                )

                # match state_values tensor dimensions with rewards tensor
                state_values = torch.squeeze(state_values)

                # logprobs, ent_loss = eval_actions(pis.squeeze(), a_mb_t_all_env[i])
                ratios = torch.exp(logprobs - old_logprobs_mb_t_all_env[i].detach())
                
                # Finding Surrogate Loss
                advantages = rewards - state_values.detach()   
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

                # final loss of clipped objective PPO
                v_loss = self.MseLoss(state_values, rewards_all_env[i])
                p_loss = - torch.min(surr1, surr2).mean()
                dist_entropy_loss = - dist_entropy_loss.clone()
                loss = vloss_coef * v_loss + ploss_coef * p_loss + entloss_coef * dist_entropy_loss
                loss_sum += loss
                vloss_sum += v_loss
           
            # x = loss_sum.mean().clone().to(torch.float64)
            self.optimizer.zero_grad()
            loss_sum.mean().backward() #TODO fix Double float tensors
            # x.backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        if configs.decayflag:
            self.scheduler.step()
        return loss_sum.mean().item(), vloss_sum.mean().item()
    

if __name__ == '__main__':
    # Debug
    from ppo_train import main
    main()