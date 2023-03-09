import sys
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from copy import deepcopy

from parameters import configs
from models.doubleActorCritic import ActorCritic
from models.dag_aggregate import aggr_obs, dag_pool



device = torch.device(configs.device)

def eval_actions(p,ix_actions):
        softmax_dist = Categorical(p)
        ret = softmax_dist.log_prob(ix_actions).reshape(-1)
        entropy = softmax_dist.entropy().mean()
        return ret, entropy


class Memory: #TODO Clean non-used vars
    def __init__(self):
        self.adj_mb = []
        self.featTask = []
        self.featMach = []
        self.state_ft = []
        self.state_fm = []
        self.candidate_mb = []
        self.mask_mb = []
        self.a_mb = [] #action_idx_task
        self.am_mb = [] #action_idx_machine
        self.reward_mb = []
        self.done_mb = []
        self.logprobs = []
        self.logprobs_m = [] 

    def clear_memory(self):
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
        
        self.policy = ActorCritic(state_dim,device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=configs.lr_agent)
    
        self.policy_old =  deepcopy(self.policy)

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=configs.decay_step_size,
                                                         gamma=configs.decay_ratio)

        self.MseLoss = torch.nn.MSELoss()
        


    def update(self,memories):
        vloss_coef = configs.vloss_coef
        ploss_coef = configs.ploss_coef
        entloss_coef = configs.entloss_coef
        
        adj_mb_t_all_env = []
        state_ft_all_env, state_fm_all_env = [], []
        candidate_mb_t_all_env = []
        mask_mb_t_all_env = []
        rewards_all_env = []
        a_mb_t_all_env = []  # task action
        am_mb_t_all_env = [] # device action
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
        
            candidate_mb_t_all_env.append(torch.stack(memories[i].candidate_mb).to(device).squeeze())
            mask_mb_t_all_env.append(torch.stack(memories[i].mask_mb).to(device).squeeze())
            a_mb_t_all_env.append(torch.stack(memories[i].a_mb).to(device).squeeze())
            am_mb_t_all_env.append(torch.stack(memories[i].am_mb).to(device).squeeze())
            old_logprobs_mb_t_all_env.append(torch.stack(memories[i].logprobs).to(device).squeeze().detach())
            old_logprobs_m_mb_t_all_env.append(torch.stack(memories[i].logprobs_m).to(device).squeeze().detach())

        
        mb_g_pool = dag_pool(configs.graph_pool_type, torch.stack(memories[0].adj_mb).to(device).shape, configs.n_tasks, device)

        # Optimize policy for K epochs:
        for _ in range(self.k_epochs):
            loss_sum = 0
            vloss_sum = 0

            for i in range(len(memories)):
                task_action, ix_action, pis, vals, log_taskprob, machine_action, mhis, valsm, log_machprob = self.policy(
                    state_ft=state_ft_all_env[i],
                    state_fm=state_fm_all_env[i],
                    candidate=candidate_mb_t_all_env[i],
                    mask=mask_mb_t_all_env[i],
                    adj=adj_mb_t_all_env[i],
                    graph_pool=mb_g_pool
                )
                logprobs, ent_loss = eval_actions(pis.squeeze(),a_mb_t_all_env[i])
                logprobs_m, ent_loss_m = eval_actions(mhis.squeeze(),am_mb_t_all_env[i])

                ratios = torch.exp(logprobs - old_logprobs_mb_t_all_env[i].detach())
                ratios_m = torch.exp(logprobs_m - old_logprobs_m_mb_t_all_env[i].detach())
                
                advantages_t = rewards_all_env[i] - vals.view(-1).detach()
                advantages_m = rewards_all_env[i] - valsm.view(-1).detach() #TODO. Note IV
                
                # Surrogate Loss and clipped objective PPO
                surr1t = ratios * advantages_t
                surr2t = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_t
                surr1m = ratios_m * advantages_m
                surr2m = torch.clamp(ratios_m, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_m

                # Final losss
                v_losst = self.MseLoss(vals.squeeze(), rewards_all_env[i])
                v_lossm = self.MseLoss(valsm.squeeze(), rewards_all_env[i])

                p_losst = - torch.min(surr1t, surr2t).mean()
                p_lossm = - torch.min(surr1m, surr2m).mean()

                # entropy
                ent_loss = - ent_loss.clone()
                ent_loss_m = - ent_loss_m.clone()

                losst = vloss_coef * v_losst + ploss_coef * p_losst + entloss_coef * ent_loss
                lossm = vloss_coef * v_lossm + ploss_coef * p_lossm + entloss_coef * ent_loss_m
                
                # Average value of each PPO
                loss_sum += (losst+lossm)/2.
                vloss_sum += (v_lossm+v_losst)/2.
           

            self.optimizer.zero_grad()
            loss_sum.mean().backward() #TODO fix Double float tensors
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        if configs.decayflag:
            self.scheduler.step()
        return loss_sum.mean().item(), vloss_sum.mean().item()
    

if __name__ == '__main__':
    # Debug
    from train_ppo import main
    main()
