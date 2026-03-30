import einops
import os
from collections import deque
import torch.nn.functional as F
import torch
from torch import nn

import copy
import cv2
import utils.net_utils as utils
from agent.networks.rgb_modules import BaseEncoder, ResnetEncoder

from agent.networks.mlp import MLP
from agent.retriever import get_retriever
import numpy as np


class ReplayBuffer:
    def __init__(self,
                 obs_shape,
                 action_dim,
                 batch_size=64,
                 max_size=int(1e5),
                 expert_size=int(1e4),
                 expert_ratio=0.3
                 ):
        self.max_size = max_size
        self.expert_max_size = expert_size
        self.expert_ratio = expert_ratio
        self.ptr = 0
        self.size = 0
        self.batch_size = batch_size
        self.current_ep_start_index=0

        # online replay buff
        self.obs = {}
        self.next_obs = {}
        self.actions = {}
        #for key, shape in obs_shape.items():

        #dtype = np.uint8 if key == "pixels" else np.float32
        self.obs["pixels"] = np.zeros((max_size, *obs_shape['pixels']), dtype=np.uint8)
        self.next_obs["pixels"] = np.zeros((max_size, *obs_shape['pixels']), dtype=np.uint8)

        for key in ['policy','retrieve']:
            self.actions[key] = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.bool8)

        self.expert_ptr = 0
        self.expert_size = 0
        self.agu_start_ptr = 0

        #expert replay buffer
        self.exp_obs = {}
        self.exp_actions = {}
        self.exp_next_obs = {}
        self.exp_obs["pixels"] = np.zeros((expert_size, *obs_shape["pixels"]), dtype=np.uint8 )
        self.exp_next_obs["pixels"] = np.zeros((expert_size, *obs_shape["pixels"]), dtype=np.uint8)
        self.exp_reward = np.zeros((expert_size, 1), dtype=np.float32)
        self.exp_done = np.zeros((expert_size, 1), dtype=np.bool8)
        for key in ['policy', 'retrieve']:
            self.exp_actions[key] = np.zeros((expert_size, action_dim), dtype=np.float32)

    def add_step(self,obs_dict,next_obs_dict,act_dict,reward,done,add=True):

        for key in self.obs.keys():
            self.obs[key][self.ptr] = obs_dict[key]
            self.next_obs[key][self.ptr] = next_obs_dict[key]


        for key in self.actions.keys():
            self.actions[key][self.ptr] = act_dict[key]

        self.reward[self.ptr] = reward
        self.done[self.ptr] = next_obs_dict["goal_achieved"]

        if add and done:
            success = next_obs_dict["goal_achieved"]
            if success:
                ep_length = self.ptr - self.current_ep_start_index + 1
                if ep_length + self.expert_size > self.expert_max_size:
                    self.save_expert_npz(f'all_retrieve_traj.npz')
                    print('=================Save expert done=========================')
                    add = False
                else:
                    self.add_expert_episode(self.current_ep_start_index, self.ptr)
                    #y
                    #print(f'=================Save expert traj current size{self.expert_size}=========================')
            self.current_ep_start_index = (self.ptr + 1) % self.max_size

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        return add


    def add_expert_episode(self,ep_start_index, ep_end_index):
        if ep_end_index >= ep_start_index:
            indices = np.arange(ep_start_index, ep_end_index + 1)
        else:
            # wrapped around the ring buffer
            indices = np.concatenate([
                np.arange(ep_start_index, self.max_size),
                np.arange(0, ep_end_index + 1)
            ])
        for idx in indices:
            exp_i = self.expert_ptr

            # copy observations
            for key in self.exp_obs.keys():
                self.exp_obs[key][exp_i] = self.obs[key][idx]
                self.exp_next_obs[key][exp_i] = self.next_obs[key][idx]

            # copy actions
            for key in self.exp_actions.keys():
                self.exp_actions[key][exp_i] = self.actions[key][idx]

            self.exp_reward[exp_i] = self.reward[idx]
            self.exp_done[exp_i] = self.done[idx]

            # advance expert pointer
            self.expert_ptr = (self.expert_ptr + 1) % self.expert_max_size
            self.expert_size = min(self.expert_size + 1, self.expert_max_size)

    def add_demo(self, traj):
        indices = np.arange(self.expert_ptr, self.expert_ptr+len(traj['actions']))

        for idx in indices:
            exp_i = self.expert_ptr

            # copy observations
            #image =  traj['observations']['pixels'][idx].transpose(2, 0, 1)

            self.exp_obs['pixels'][exp_i] =  traj['observations']['pixels'][idx].transpose(2, 0, 1) #cv2.resize(image, (84, 84), interpolation=cv2.INTER_AREA)

            # copy actions
            self.exp_actions['policy'][exp_i] = traj['actions'][idx]
            self.exp_actions['retrieve'][exp_i] = traj['actions'][idx]

            # advance expert pointer
            self.expert_ptr = (self.expert_ptr + 1) % self.expert_max_size
            self.expert_size = min(self.expert_size + 1, self.expert_max_size)
            self.agu_start_ptr = self.expert_ptr

    def save_expert_npz(self, filename="expert_buffer.npz"):
        # 创建一个要保存的字典，把所有数据塞进去
        save_dict = {
            "size": self.expert_size,
            "reward": self.exp_reward[:self.expert_size],
            "done": self.exp_done[:self.expert_size],
        }

        for k, v in self.exp_obs.items():
            save_dict[f"obs_{k}"] = v[:self.expert_size]

        for k, v in self.exp_actions.items():
            save_dict[f"action_{k}"] = v[:self.expert_size]

        for k, v in self.exp_next_obs.items():
            save_dict[f"next_obs_{k}"] = v[:self.expert_size]

        # 执行保存
        np.savez_compressed(filename, **save_dict)
        print(f"[ReplayBuffer] Expert dataset saved to {filename}")

    def sample_buff(self, ):
        n_exp = int(self.expert_ratio * self.batch_size)
        if n_exp>0 and self.expert_size>self.agu_start_ptr:
            exp_idx = np.random.randint(self.agu_start_ptr, self.expert_size , size=n_exp)
        else:
            exp_idx = []

        n_onl = self.batch_size - len(exp_idx)

        onl_idx = np.random.randint(0, self.size, size=n_onl)

        obs = {}
        next_obs = {}
        action = {}
        for key in self.obs.keys():
            onl_part = self.obs[key][onl_idx]
            onl_next = self.next_obs[key][onl_idx ]

            exp_part = self.exp_obs[key][exp_idx] if len(exp_idx) > 0 else []
            exp_next = self.exp_next_obs[key][exp_idx] if len(exp_idx) > 0 else []

            combined = np.concatenate([exp_part, onl_part], axis=0) if len(exp_idx) > 0 else onl_part
            combined_next = np.concatenate([exp_next, onl_next], axis=0) if len(exp_idx) > 0 else onl_next

            obs[key] = torch.as_tensor(combined, dtype=torch.float32)
            next_obs[key] = torch.as_tensor(combined_next, dtype=torch.float32)

        for key in self.actions.keys():
            onl_act = self.actions[key][onl_idx]
            exp_act = self.exp_actions[key][exp_idx] if len(exp_idx) > 0 else []
            act_combined = np.concatenate([exp_act, onl_act], axis=0) if len(exp_idx) > 0 else onl_act

            action[key] = torch.as_tensor(act_combined, dtype=torch.float32)

        #reward
        onl_r = self.reward[onl_idx]
        exp_r = self.exp_reward[exp_idx] if len(exp_idx) > 0 else []
        rewards = np.concatenate([exp_r, onl_r], axis=0) if len(exp_idx) > 0 else onl_r
        reward = torch.as_tensor(rewards, dtype=torch.float32)

        onl_d = self.done[onl_idx]
        exp_d = self.exp_done[exp_idx] if len(exp_idx) > 0 else []
        dones = np.concatenate([exp_d, onl_d], axis=0) if len(exp_idx) > 0 else onl_d
        done = torch.as_tensor(dones, dtype=torch.float32)

        return obs,action,reward,done,next_obs

    def sample_exp(self, ):
        # draw from expert buffer (FIFO)
        exp_idx = np.random.randint(0, self.expert_size, size=self.batch_size)
        obs = {}
        action = {}

        exp_part = self.exp_obs['pixels'][exp_idx]
        obs['pixels'] = torch.as_tensor(exp_part, dtype=torch.float32)

        for key in self.actions.keys():
            exp_act = self.exp_actions[key][exp_idx] if len(exp_idx) > 0 else []
            action[key] = torch.as_tensor(exp_act, dtype=torch.float32)

        return obs,action


class Critic(nn.Module):
    def __init__(self, repr_dim, action_dim, feature_dim=None):
        super().__init__()

        feature_dim = 50
        hidden_dim = repr_dim
        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim),
                                   nn.Tanh())
        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1))
        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1))
        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)
        return q1, q2

class Actor(nn.Module):
    def __init__(self, hidden_dim, action_dim,):
        super().__init__()

        self.policy = nn.Sequential(nn.Linear(hidden_dim, hidden_dim, bias=True),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim, bias=True),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, action_dim, bias=True),
                                    nn.Tanh()
                                    )
        self.apply(utils.weight_init)

    def forward(self, obs, std):
        mu = self.policy(obs)
        std = torch.ones_like(mu) * std
        dist = utils.TruncatedNormal(mu, std)
        return dist


class RegAgent:
    def __init__(
        self,
        name,
        obs_shape,
        action_shape,
        device,
        lr,
        hidden_dim,
        critic_target_tau,
        stddev_sch,
        use_tb,
        obs_type,
        encoder_type,
        pixel_keys,
        feature_key,
        train_encoder,
        q_filter_act,

        gamma,
        retrieve,
        max_episode_len,
        stddev_clip,

        bc_type,
        retrieve_len,
        re_history_len,

        replay_warmup,
        policy_freq,
        batch_size,

        #debug
        enc_up_critic,
        bc_weight_schedule,
        pretrain_encoder,
        bc_enable,
        rl_enable,
        reward_type,
        add_expert,
        enc_update,
        expert_ratio,
        critic_scale,
        success_scale,
        lamda
    ):
        self.device = device
        self.lr = lr
        self.gamma = gamma
        self.hidden_dim = hidden_dim
        self.use_tb = use_tb
        self.obs_type = obs_type
        self.encoder_type = encoder_type

        self.train_encoder = train_encoder
        self.q_filter_act = q_filter_act

        self.critic_target_tau = critic_target_tau
        self.stddev_sch = stddev_sch
        self.stddev_clip = stddev_clip
        self.stddev = utils.schedule(self.stddev_sch,1)
        self.bc_type = bc_type
        if self.bc_type == 'lamda':
            self.lamda = lamda
            self.q_filter = False
        elif self.bc_type == 'q_filter':
            self.q_filter = True
            self.lamda = 0
        elif self.bc_type == 'linear':
            self.q_filter = False
            self.lamda = 0



        self.retrieve = retrieve
        self.critic_scale = critic_scale
        self.success_scale = success_scale


        self.replay_warmup = replay_warmup
        self.add_expert = add_expert

        self.enc_update = enc_update
        if self.enc_update>0:
            self.enc_update = 1 if bc_enable else 2
        self.enc_up_critic = enc_up_critic

        self.bc_enable = bc_enable
        self.rl_enable = rl_enable
        self.bc_weight_schedule = bc_weight_schedule
        if rl_enable:buffer_reset
            self.policy_freq = policy_freq  # TD3 default, or pass in
            if not bc_enable:
                self.bc_weight_schedule = 'linear(0,0,400)'
        else:
            self.policy_freq = 1
            self.bc_weight_schedule = 'linear(1,1,400)'

        if not add_expert:
            expert_ratio = 0


        # actor parameters
        self._act_dim = action_shape[0]
        self.buff = ReplayBuffer(obs_shape, action_shape[0],expert_ratio=expert_ratio,batch_size=batch_size, expert_size = self.replay_warmup)

        self.re_history_len = re_history_len
        self.retrieve_len = retrieve_len
        self.retrieve_context = None

        #================================for debug=================================
        self.repr_dim = 256
        self.reward_type = reward_type
        self.reward_scale = 1
        #===============================================================================

        self.retriver = get_retriever(
            retrieve_key='DINO',
            state_num=10,
            metric='l2',
            traj_metric='sdtw',  # 'ot','sdtw'
            re_history_len=re_history_len,
            retrieve_len=retrieve_len,
        )

        # keys
        if obs_type == "pixels":
            self.pixel_keys = pixel_keys
        else:
            self.feature_key = feature_key

        #self.max_episode_len = max_episode_len

        self.train_encoder = self.train_encoder
        self.update_cnt = 0
        self.q_filter_act_cnt = []

        # observation params
        if obs_type == "pixels":
            obs_shape = obs_shape[self.pixel_keys[0]]
        else:
            obs_shape = obs_shape[self.feature_key]

        # Track model size
        model_size = 0

        # encoder
        if obs_type == "pixels":
            if self.encoder_type == "base":
                self.encoder = BaseEncoder(obs_shape).to(device)
                self.repr_dim = self.encoder.repr_dim
                model_size += sum(
                    p.numel() for p in self.encoder.parameters() if p.requires_grad
                )
            elif self.encoder_type == "resnet":
                self.encoder = ResnetEncoder(
                    obs_shape,
                    self.repr_dim,
                    pretrained=pretrain_encoder,
                    freeze=(not self.train_encoder),
                    language_fusion="none",
                ).to(device)
                model_size += sum(
                    p.numel() for p in self.encoder.parameters() if p.requires_grad
                )
                #self.repr_dim = 512
            elif self.encoder_type == "patch":
                pass
        else:
            self.encoder = MLP(obs_shape[0], hidden_channels=[512, 512]).to(device)
            model_size += sum(
                p.numel() for p in self.encoder.parameters() if p.requires_grad
            )
            self.repr_dim = 512
        #self.aug_DrQv2 = utils.RandomShiftsAug(pad=4)


        # actor
        action_dim = self._act_dim
        self.actor = Actor(
            self.hidden_dim,
            action_dim,
        ).to(device)
        model_size += sum(p.numel() for p in self.actor.parameters() if p.requires_grad)
        self.actor_opt = torch.optim.AdamW(
            self.actor.parameters(), lr=lr, weight_decay=1e-4
        )

        if rl_enable:
            self.critic = Critic(
                self.hidden_dim,
                action_dim,
            ).to(device)
            model_size += sum(p.numel() for p in self.critic.parameters() if p.requires_grad)

            self.critic_opt = torch.optim.AdamW(
                self.critic.parameters(), lr=lr, weight_decay=1e-4
            )
            self.critic_target = Critic(
            self.hidden_dim,
            action_dim,
            ).to(self.device)
            self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        # encoder
        if self.train_encoder:
            params = list(self.encoder.parameters())
            self.encoder_opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

        self.stats = {
            "actions": {
                "min": 0,
                "max": 1,
            },
        }
        self.preprocess = {
            "actions": lambda x: (x - self.stats["actions"]["min"])
                                 / (self.stats["actions"]["max"] - self.stats["actions"]["min"] + 1e-5),# useless
        }
        print(f'BC regularization: {self.bc_enable}, \n'
              f'RL regularization: {self.rl_enable},\n'
              f'Add experience: {self.add_expert}')

        self.train()

    def __repr__(self):
        return "reg"

    def train(self, training=True):
        self.training = training
        if training:
            if self.train_encoder:
                self.encoder.train(training)
            else:
                self.encoder.eval()
            self.actor.train(training)
            if self.rl_enable:
                self.critic.train(training)
        else:
            self.encoder.eval()
            self.actor.eval()
            if self.rl_enable:
                self.critic.eval()

    def buffer_reset(self,obs=None):
        if self.obs_type == "pixels":
            self.observation_buffer = {}
            for key in self.pixel_keys:
                self.observation_buffer[key] = deque(maxlen=self.re_history_len)

        if obs is not None:
            for key in self.pixel_keys:
                self.observation_buffer[key].append(
                    obs[key]
                )
            self.retrieve_context = self.get_retrieve_act()

    def clear_buffers(self):
        del self.observation_buffer


    def reinit_optimizers(self):
        if self.train_encoder:
            params = list(self.encoder.parameters())
            self.encoder_opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=1e-4)
        self.actor_opt = torch.optim.AdamW(
            self.actor.parameters(), lr=self.lr, weight_decay=1e-4
        )



    def act(self, obs,retrieve_only=False, eval_mode=False,**kwargs):
        """
        Selects an action using a Q-filter for both evaluation and training,
        optimized with torch.no_grad() for performance.
        """
        # 1. Pre-processing
        obs_tensor = utils.to_torch(obs, device=self.device)
        if retrieve_only:
            action = self.retrieve_context['retrieve_action']
            return action

        if (self.update_cnt < self.replay_warmup) and (not eval_mode):
            action = self.retrieve_context['retrieve_action']
            return action
        elif self.update_cnt == self.replay_warmup:
            self.retrieve = False
            self.buff.save_expert_npz()
            print("============================Replay warmup finished==============================")

        with torch.no_grad():
            # 2. Feature Encoding
            obs_feature = self.obs_encode(obs_tensor)

            # 3. Initial Action Selection (Policy-driven)
            action_dist = self.actor(obs_feature, self.stddev)

            if eval_mode:
                action = action_dist.mean
            else:
                # Stochastic sampling for training
                action = action_dist.sample(clip=None)

        # 5. Conversion to Numpy
        return action.cpu().numpy().squeeze()

    def update_critic(self,
                      obs,
                      action,
                      reward,
                      dones,
                      next_obs,
                      ):

        B = next_obs.shape[0]
        update_encoder = self.enc_update == 2

        if not update_encoder:
            obs = obs.detach()

        with torch.no_grad():
            dist = self.actor(next_obs, self.stddev)
            next_obs = next_obs.reshape(B, -1)
            next_action = dist.sample(clip=self.stddev_clip).reshape(B, -1)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - dones) * self.gamma  * target_V)

        obs = obs.reshape(B, -1)
        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)
        critic_loss = self.critic_scale * critic_loss

        # optimize encoder and critic
        if update_encoder:
            self.encoder_zero_grad()

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()

        self.critic_opt.step()
        if update_encoder:
            self.encoder_step()

        return critic_loss

    def update_actor(self, obs,exp_state_act,expert_action,bc_sche):
        loss = {}
        rl_loss = 0
        B = obs.shape[0]
        update_encoder = self.enc_update == 1
        if not update_encoder:
            obs = obs.detach()
            exp_state_act = exp_state_act.detach() if exp_state_act is not None else None

        if self.rl_enable:
            rl_dist = self.actor(obs, self.stddev)
            action = rl_dist.sample(clip=self.stddev_clip).reshape(B,-1)
            Q1, Q2 = self.critic(obs.reshape(B, -1), action)
            Q = torch.min(Q1, Q2)

            loss['q_mean'] = Q.mean()
            rl_loss = -Q.mean()

        if self.bc_enable and (exp_state_act is not None):
            bc_dist = self.actor(exp_state_act, self.stddev)
            log_prob_bc = bc_dist.log_prob(expert_action).reshape(B,-1)
            bc_loss = -log_prob_bc.mean()

            loss['bc_loss'] = bc_loss

            if self.q_filter and bc_sche < 1:
                with torch.no_grad():
                    stddev = utils.schedule(self.stddev_sch, self.replay_warmup - 10)
                    dist_qf = self.actor_bc(obs.clone(), stddev)
                    action_qf = dist_qf.mean
                    Q1_qf, Q2_qf = self.critic(obs.clone(), action_qf)
                    Q_qf = torch.min(Q1_qf, Q2_qf)
                    bc_weight = (Q_qf > Q).float().mean().detach()
                    q_scale = Q.abs().mean().detach() + 1e-6
                    bc_scale = bc_loss.abs().detach() + 1e-6
                    rl_weight = (1.1 - bc_weight) * ( bc_scale /q_scale )
                actor_loss = rl_loss * rl_weight + bc_loss * bc_weight
            elif self.lamda > 0 and bc_sche < 1:
                with torch.no_grad():
                    q_scale = Q.abs().mean().detach() + 1e-6
                    bc_sacle = bc_loss.abs().detach() + 1e-6
                    bc_weight = self.lamda * ( bc_sacle /q_scale ) # actually rl_weight
                actor_loss = rl_loss * bc_weight+ bc_loss
            else:
                bc_weight =  bc_sche
                actor_loss = rl_loss * (1 - bc_weight) + bc_loss * bc_weight
        else:
            bc_weight = 0
            actor_loss = rl_loss
        loss['bc_weight'] = bc_weight


        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)

        if update_encoder:
            self.encoder_zero_grad()

        actor_loss.backward()
        self.actor_opt.step()

        if update_encoder:
            self.encoder_step()

        return loss

    def encoder_zero_grad(self):
        if self.enc_update>0:
            self.encoder_opt.zero_grad(set_to_none=True)

    def encoder_step(self):
        if self.enc_update>0:
            self.encoder_opt.step()

    def obs_encode(self,obs):
        if obs[self.obs_type].ndim ==3:
            obs[self.obs_type] = obs[self.obs_type].unsqueeze(0)

        if obs[self.obs_type].ndim ==4:
            T=1
            B = obs[self.obs_type].shape[0]
        elif obs[self.obs_type].ndim  ==5:
            B = obs[self.obs_type].shape[0]
            T = obs[self.obs_type].shape[1]

        # features
        if self.obs_type == "pixels":
            features = []
            #for key in self.pixel_keys:
            pixel = (obs[self.obs_type] / 255.0).float().to(self.device)
            #shape = pixel.shape
            # rearrange
            if obs[self.obs_type].ndim ==5:
                pixel = einops.rearrange(pixel, "b t c h w -> (b t) c h w",b=B, t=T)

            if self.enc_update>0:
                pixel = self.encoder(pixel, lang=None)
            else:
                with torch.no_grad():
                    pixel = self.encoder(pixel, lang=None)
            if obs[self.obs_type].ndim == 5:
                pixel = einops.rearrange(pixel, "(b t) d -> b t d", b=B, t=T)
            features.append(pixel)
        else:
            features = obs[self.feature_key].float()
            if self.train_encoder:
                features = self.encoder(features)
            else:
                with torch.no_grad():
                    features = self.encoder(features)
        features = features[0]

        return features

    def update(self,):
        metrics = dict()
        bc_weight = utils.schedule(self.bc_weight_schedule, self.update_cnt)
        self.stddev = utils.schedule(self.stddev_sch, self.update_cnt)
        metrics['std'] = self.stddev

        if self.update_cnt> self.replay_warmup:
            self.enc_update = 0
        metrics['expert_buff'] = self.buff.expert_size
        metrics['enc_train'] = self.enc_update
        if self.buff.size< self.buff.batch_size:
            metrics['critic_loss'] = 0
            metrics['bc_loss'] = 0
            metrics['q_mean'] = 0
            metrics['bc_weight'] = bc_weight
            return metrics
        self.update_cnt += 1
        obs_dict, act_dict, reward, done, next_obs_dict = self.buff.sample_buff()
        exp_obs_dict, exp_act_dict= self.buff.sample_exp()

        reward = reward.to(device=self.device)
        done = done.to(device=self.device)


        policy_action = act_dict['policy'].to(device=self.device)
        expert_action = exp_act_dict['policy'].to(device=self.device)

        if self.update_cnt == self.replay_warmup:
            #self.add_expert = False
            if self.q_filter:
                self.actor_bc = copy.deepcopy(self.actor)
                for param in self.actor_bc.parameters():
                    param.required_grad = False

        # update critic
        if self.rl_enable:
            state = self.obs_encode(obs_dict).to(device=self.device) #B,256
            next_state = self.obs_encode(next_obs_dict).to(device=self.device).detach()
            critic_loss = self.update_critic(state, policy_action, reward, done, next_state)
            metrics['critic_loss'] = critic_loss

        if self.update_cnt % self.policy_freq == 0:
            state_act = self.obs_encode(obs_dict).to(device=self.device)  # B,256
            if exp_obs_dict is not None:
                exp_state_act = self.obs_encode(exp_obs_dict).to(device=self.device)  # B,256

            loss = self.update_actor(state_act,exp_state_act,expert_action,bc_weight)
            for key in loss.keys():
                metrics[key] = loss[key]
            if self.rl_enable:
                utils.soft_update_params(self.critic,self.critic_target,self.critic_target_tau)

        return metrics

    def init_demos(self,traj,skip=None,eval=False):
        if skip is None:
            self.demo = traj
        else:
            #self.demo = []
            traj_len = len(traj['actions'])
            new_len = traj_len // skip
            idx = np.arange(0, new_len * skip, skip) #idx = np.linspace(0, traj_len - 1, new_len).astype(int)
            save_traj = {}
            save_action = traj['actions'][idx,]
            save_traj['actions'] =  self.preprocess["actions"](save_action) #()
            for key in traj['observations'].keys():
                if key in self.preprocess.keys():
                    save_traj[key] = self.preprocess[key](traj['observations'][key][idx])
                else:
                    save_traj[key] = traj['observations'][key][idx]
            self.demo=save_traj

            if not eval:
                traj_len = len(traj['actions'])
                if traj_len>0:
                    self.buff.add_demo(traj)
                else:
                    print('Demo is not valid')

        return self.demo

    def update_obs_and_retrieve(self, obs_dict):
        self.observation_buffer[self.obs_type].append(obs_dict[self.obs_type])
        self.retrieve_context = self.get_retrieve_act()
        return self.retrieve_context

    def get_reward(self,):
        self.retrieve_context = self.get_retrieve_act()
        norm_best_dis = self.retrieve_context['norm_best_dist']
        retrieve_action = self.retrieve_context['retrieve_action']
        best_dis = self.retrieve_context['best_dist']

        reward_dict = {}
        reward_dict['norm_best_dist'] = -norm_best_dis
        reward_dict['best_dist'] = -best_dis
        reward = -norm_best_dis

        return reward,retrieve_action,reward_dict


    def get_retrieve_act(self):
        context, best_dist,retrieve_state_idx_s,retrieve_state_idx,path_len = self.get_retrieve_observations()
        best_dist = best_dist/(retrieve_state_idx-retrieve_state_idx_s+1)
        norm_best_dist = best_dist/path_len
        retrieve_index = 0
        action = context['action'][0]
        self.retrieve_context = {
            'norm_best_dist':norm_best_dist,
            'best_dist':best_dist,
            'retrieve_index':retrieve_index,
            'retrieve_context':context,
            'retrieve_action':action,
            'index_in_demo':retrieve_state_idx
        }

        return self.retrieve_context


    def get_retrieve_observations(self,ob_history =None):
        if ob_history is None:
            ob_history = list(self.observation_buffer[self.pixel_keys[0]])
        state_traj = []
        for state_img in ob_history:
            state_traj.append(self.retriver.state_encode(state_img))
        sample_retrieve_traj = state_traj

        if self.retriver.state_num == 'none':
            retrieve_traj_idx, retrieve_state_idx_s, retrieve_state_idx_end, best_dist, path_len = self.retriver.get_traj_index_from_subset_traj(
                sample_retrieve_traj,
                [((0,len(self.demo[0]['action'])),1)],
                self.demo)
        else:
            state_subset = self.retriver.get_state_subset_from_task(
                sample_retrieve_traj,
                self.demo,
            )
            if self.retriver.traj_metric=='none':
                (retrieve_traj_idx, state_id), score = state_subset[0]
                retrieve_state_idx_s = state_id
                retrieve_state_idx_end = state_id
                best_dist = score
                path_len = 1
            else:

                refine_sub = self.refine_state_subset(state_subset)

                retrieve_state_idx_s,retrieve_state_idx_end,best_dist,path_len = self.retriver.get_traj_index_from_subset_traj(
                    sample_retrieve_traj,
                    refine_sub,
                    self.demo)

        end_idx = min(retrieve_state_idx_end+self.retrieve_len,len(self.demo['actions']))

        retrived_traj = self.demo
        retrieved_obs = retrived_traj[self.pixel_keys[0]][retrieve_state_idx_end:end_idx]
        retrieved_act = retrived_traj['actions'][retrieve_state_idx_end:end_idx]

        retrieved_feature = retrived_traj['retrieve_feature'][retrieve_state_idx_end:end_idx]
        context = {
            'obs':retrieved_obs,
            'action':retrieved_act,
            'retrieve_feature':retrieved_feature,
        }

        return context,best_dist,retrieve_state_idx_s,retrieve_state_idx_end,path_len

    def add_buffer(self,
                   observation,
                   next_observation,
                   done,
                   policy_action,
                   retrive_reward,
                   retrieve_action,
                   success=False):
        obs = observation.copy()
        next_obs = next_observation.copy()
        next_obs["goal_achieved"] = success
        reward =retrive_reward

        act_dict={
            'policy':self.preprocess['actions'](policy_action),
            'retrieve':self.preprocess['actions'](retrieve_action)
        }
        self.add_expert = self.buff.add_step(obs,next_obs,act_dict,reward,done,self.retrieve)


    def refine_state_subset(self,state_subset,time_step=None):
        max_pair = max(state_subset, key=lambda x: x[1])

        # Return in the original format as a list
        return [max_pair]

    def save_snapshot(self, save_dir):
        if self.update_cnt<10:
            return
        keys_to_save = ["actor",
                        "encoder",
                        "actor_opt",
                        "encoder_opt",
                        ]
        if self.rl_enable:
            keys_to_save = keys_to_save + ["critic", ]  # critic_opt
        payload = {k: self.__dict__[k].state_dict() for k in keys_to_save}
        others = [
            #"max_episode_len",
            'update_cnt'
        ]
        payload.update({k: self.__dict__[k] for k in others})


        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        filename = f"snapshot.pt"
        path = os.path.join(save_dir, filename)
        torch.save(payload, path)
        print(f"[Model] snapshot saved to {path}")

        return payload

    def load_snapshot(self, payload, eval=False, load_opt=False):
        # Define keys to load
        model_keys = ["actor",  "encoder",]
        opt_keys = ["actor_opt",  "encoder_opt", ]

        if self.rl_enable:
            model_keys = model_keys + ["critic", ]  # critic_opt

        # Load model weights
        for key in model_keys:
            if key in payload and key in self.__dict__:
                self.__dict__[key].load_state_dict(payload[key])
                print(f"  ✓ Loaded {key}")
            else:
                print(f"  ⚠️ Missing {key} in  model")

        if load_opt:
            for key in opt_keys:
                if key in payload and key in self.__dict__:
                    self.__dict__[key].load_state_dict(payload[key])
                    print(f"  ✓ Loaded optimizer {key}")

        else:
            self.reinit_optimizers()
        # Switch to eval mode if requested
        if eval:
            for key in model_keys:
                if key in self.__dict__:
                    self.__dict__[key].eval()

            print("Snapshot successfully loaded.")
