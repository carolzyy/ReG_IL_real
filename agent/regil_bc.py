import einops
import math
from collections import deque
import torch.nn.functional as F
import torch
from torch import nn

import os
import copy
import utils.net_utils as utils
from agent.networks.rgb_modules import BaseEncoder, ResnetEncoder
from agent.networks.mlp import MLP
from agent.retriever import get_retriever
import numpy as np

from utils.read_data import ReplayBuffer


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


class RegBCAgent:
    def __init__(
        self,
        name,
        obs_shape,
        action_shape,
        device,
        lr,
        hidden_dim,
        stddev_sch,
        use_tb,
        obs_type,
        encoder_type,
        pixel_keys,
        feature_key,
        train_encoder,

        retrieve,
        max_episode_len,
        stddev_clip,

        retrieve_len,
        re_history_len,

        replay_warmup,
        batch_size,

        #debug
        pretrain_encoder,
        add_expert,
    ):
        self.device = device
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.use_tb = use_tb
        self.obs_type = obs_type
        self.encoder_type = encoder_type

        self.train_encoder = train_encoder
        self.name = name

        self.stddev_sch = stddev_sch
        self.stddev_clip = stddev_clip
        self.stddev = utils.schedule(self.stddev_sch,1)


        self.retrieve = retrieve


        self.replay_warmup = replay_warmup
        self.add_expert = add_expert

        self.enc_update = 1




        # actor parameters
        self._act_dim = action_shape[0]
        self.buff = ReplayBuffer(obs_shape,
                                 action_shape[0],
                                 batch_size=batch_size,
                                 expert_size=self.replay_warmup,)

        self.re_history_len = re_history_len
        self.retrieve_len = retrieve_len
        self.retrieve_context = None

        #================================for debug=================================
        self.repr_dim = 256
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

        self.max_episode_len = max_episode_len

        self.train_encoder = self.train_encoder
        self.update_cnt = 0

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
        print(f'BC Agent, Retrieve: {self.retrieve},Add experience: {self.add_expert}')

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
        else:
            self.encoder.eval()
            self.actor.eval()

    def buffer_reset(self,obs=None):
        if self.obs_type == "pixels":
            self.observation_buffer = {}
            for key in self.pixel_keys:
                self.observation_buffer[key] = deque(maxlen=self.re_history_len)
        else:
            self.observation_buffer = deque(maxlen=self.re_history_len)


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



    def act(self, obs, eval_mode=False,**kwargs):
        """
        Selects an action using a Q-filter for both evaluation and training,
        optimized with torch.no_grad() for performance.
        """
        # 1. Pre-processing
        obs_tensor = utils.to_torch(obs, device=self.device)
        if (self.update_cnt < self.replay_warmup) and (not eval_mode):
            if getattr(self, 'retrieve', False):
                action = self.retrieve_context['retrieve_action']
                return action
        elif (not eval_mode):
            self.retrieve = False

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


    def update_actor(self, exp_state_act,expert_action):
        loss = {}
        B = exp_state_act.shape[0]
        update_encoder = self.enc_update == 1
        loss['rl_loss'] =0
        if not update_encoder:
            exp_state_act = exp_state_act.detach() if exp_state_act is not None else None


        if exp_state_act is not None:
            bc_dist = self.actor(exp_state_act, self.stddev)
            log_prob_bc = bc_dist.log_prob(expert_action).reshape(B,-1)
            bc_loss = -log_prob_bc.mean()

            loss['bc_loss'] = bc_loss
        loss['bc_weight'] = 1
        actor_loss = bc_loss


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
        self.stddev = utils.schedule(self.stddev_sch, self.update_cnt)
        metrics['std'] = self.stddev

        if self.update_cnt> self.replay_warmup:
            self.enc_update = 0
        metrics['expert_buff'] = self.buff.expert_size
        metrics['enc_train'] = self.enc_update
        self.update_cnt += 1
        exp_obs_dict, exp_act_dict= self.buff.sample_exp()
        expert_action = exp_act_dict['policy'].to(device=self.device)
        if exp_obs_dict is not None:
            exp_state_act = self.obs_encode(exp_obs_dict).to(device=self.device)  # B,256

        loss = self.update_actor(exp_state_act,expert_action)
        for key in loss.keys():
            metrics[key] = loss[key]
        return metrics

    def init_demos(self,traj,skip=None,eval=False):
        if skip is None:
            self.demo = traj
        else:
            traj_len = len(traj['actions'])
            new_len = traj_len // skip
            idx = np.arange(0, new_len * skip, skip)  # idx = np.linspace(0, traj_len - 1, new_len).astype(int)
            save_traj = {}
            save_action = traj['actions'][idx,]
            save_traj['actions'] = self.preprocess["actions"](save_action)  # ()
            for key in traj['observations'].keys():
                if key in self.preprocess.keys():
                    save_traj[key] = self.preprocess[key](traj['observations'][key][idx])
                else:
                    save_traj[key] = traj['observations'][key][idx]
            self.demo = save_traj

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

    def get_reward(self, next_timestep):
        next_obs = next_timestep.observation
        self.retrieve_context = self.get_retrieve_act()
        norm_best_dis = self.retrieve_context['norm_best_dist']
        retrieve_action = self.retrieve_context['retrieve_action']
        best_dis = self.retrieve_context['best_dist']

        reward_dict = {}
        reward_dict['norm_best_dist'] = -norm_best_dis
        reward_dict['best_dist'] = -best_dis

        success = next_obs["goal_achieved"]
        reward_dict['binary'] = float(success)

        reward_dict['gt_reward'] = next_timestep.reward

        reward = reward_dict['norm_best_dist']

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

                retrieve_traj_idx, retrieve_state_idx_s,retrieve_state_idx_end,best_dist,path_len = self.retriver.get_traj_index_from_subset_traj(
                    sample_retrieve_traj,
                    refine_sub,
                    self.demo)

        end_idx = min(retrieve_state_idx_end+self.retrieve_len,len(self.demo[retrieve_traj_idx]['action']))

        retrived_traj = self.demo[retrieve_traj_idx]
        retrieved_obs = retrived_traj[self.pixel_keys[0]][retrieve_state_idx_end:end_idx]
        retrieved_act = retrived_traj['action'][retrieve_state_idx_end:end_idx]

        retrieved_feature = retrived_traj['retrieve_feature'][retrieve_state_idx_end:end_idx]
        context = {
            'obs':retrieved_obs,
            'action':retrieved_act,
            'retrieve_feature':retrieved_feature,
        }

        return context,best_dist,retrieve_state_idx_s,retrieve_state_idx_end,path_len

    def add_buffer(self,time_step, next_time_step,retrive_reward, retrieve_action):
        obs = time_step.observation.copy()
        next_obs = next_time_step.observation.copy()
        reward =retrive_reward
        done = next_time_step.last()
        policy_action =next_time_step.action

        act_dict={
            'policy':self.preprocess['actions'](policy_action),
            'retrieve':self.preprocess['actions'](retrieve_action)
        }
        self.add_expert = self.buff.add_step(obs,next_obs,act_dict,reward,done,self.retrieve)


    def refine_state_subset(self,state_subset,time_step=None):
        refine_subset = {}
        for (traj_id, state_id), score in state_subset:
            # If this traj is new OR this state_id is larger, replace it
            if traj_id not in refine_subset or state_id > refine_subset[traj_id][0][1]:
                refine_subset[traj_id] = ((traj_id, state_id), score)

        # Return in the original format as a list
        return list(refine_subset.values())

    def save_snapshot(self, save_dir):
        if self.update_cnt<10:
            return
        keys_to_save = ["actor",
                        "encoder",
                        "actor_opt",
                        "encoder_opt",
                        ]
        payload = {k: self.__dict__[k].state_dict() for k in keys_to_save}
        others = [
            "max_episode_len",
        ]
        payload.update({k: self.__dict__[k] for k in others})


        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        filename = f"bc_model.pt"
        path = os.path.join(save_dir, filename)
        torch.save(payload, path)
        print(f"[Model] snapshot saved to {path}")

        return payload

    def load_snapshot(self, payload, eval=False, load_opt=False):
        # Define keys to load
        model_keys = ["actor",  "encoder",]
        opt_keys = ["actor_opt",  "encoder_opt", ]

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
