import cv2
import random
import numpy as np
import pickle as pkl
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch.utils.data import IterableDataset

class BCDataset(IterableDataset):
    def __init__(
        self,
        path,
        tasks,
        num_demos_per_task,
        obs_type,
        history,
        history_len,
        prompt,
        temporal_agg,
        num_queries,
        img_size,
        intermediate_goal_step,
        store_actions,
        pixel_keys,
        subsample=None,
    ):
        self._obs_type = obs_type
        self._prompt = prompt
        self._history = history
        self._history_len = history_len if history else 1
        self._img_size = img_size
        self._intermediate_goal_step = intermediate_goal_step
        self._store_actions = store_actions
        self._pixel_keys = pixel_keys

        # temporal aggregation
        self._temporal_agg = temporal_agg
        self._num_queries = num_queries

        # get data paths
        self._paths = []
        self._paths.extend([Path(path) / f"{task}.npy" for task in tasks])

        paths = {}
        idx = 0
        for path in self._paths:
            paths[idx] = path
            idx += 1
        del self._paths
        self._paths = paths

        # store actions
        if self._store_actions:
            self.actions = []

        # read data
        self._episodes = {}
        self._max_episode_len = 0
        self._max_state_dim = 0
        self._num_samples = 0
        self.stats = {}
        min_stat, max_stat = None, None
        min_act, max_act = None, None
        for _path_idx in self._paths:
            print(f"Loading {str(self._paths[_path_idx])}")
            # read
            data = np.load(self._paths[_path_idx],allow_pickle=True).item()
            observations =[data["observations"]]

            # store
            self._episodes[_path_idx] = []
            for i in range(min(num_demos_per_task, len(observations))):
                # compute actions
                # absolute actions
                action = data["actions"]
                self.stats = {
                    'raw_max': action.max(axis=0),
                    'raw_min': action.min(axis=0),
                }
                act_range = self.stats['raw_max'][:3] - self.stats['raw_min'][:3]
                action_xyz = 2 * (action[..., :3] - self.stats['raw_min'][:3]) / (act_range + 1e-8) - 1
                action_gripper = np.where(action[..., 3:] > 0.5, 1.0, -1.0)
                action_processed = np.concatenate([action_xyz, action_gripper], axis=-1)
                if len(action_processed) == 0:
                    print('The action length is 0')
                    break

                # subsample
                if subsample is not None:
                    for key in observations[i].keys():
                        observations[i][key] = observations[i][key][::subsample]
                    action_processed = action_processed[::subsample]

                # Repeat last dimension of each observation for history_len times
                for key in observations[i].keys():
                    observations[i][key] = np.concatenate(
                        [
                            observations[i][key],
                            [observations[i][key][-1]] * self._history_len,
                        ],
                        axis=0,
                    )
                # Repeat last action for history_len times
                remaining_actions = action_processed[-1]
                actions = np.concatenate(
                    [
                        action_processed,
                        [remaining_actions] * self._history_len,
                    ],
                    axis=0,
                )

                # store
                episode = dict(
                    observation=observations[i],
                    action=actions,
                )
                self._episodes[_path_idx].append(episode)
                self._max_episode_len = max(
                    self._max_episode_len,
                    (
                        len(observations[i])
                        if not isinstance(observations[i], dict)
                        else len(observations[i][self._pixel_keys[0]])
                    ),
                )
                #self._max_state_dim = 7
                self._num_samples += len(observations[i][self._pixel_keys[0]])

                # max, min action
                if min_act is None:
                    min_act = np.min(actions, axis=0)
                    max_act = np.max(actions, axis=0)
                else:
                    min_act = np.minimum(min_act, np.min(actions, axis=0))
                    max_act = np.maximum(max_act, np.max(actions, axis=0))

                # store actions
                if self._store_actions:
                    self.actions.append(actions)
        self.stats["actions"] = {
                "min": min_act,
                "max": max_act,
        }
        self.preprocess = {
            "actions": lambda x: (x - self.stats["actions"]["min"])
            / (self.stats["actions"]["max"] - self.stats["actions"]["min"] + 1e-5),

        }

        # augmentation
        self.aug = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomCrop(self._img_size, padding=4),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                transforms.ToTensor(),
            ]
        )

        # Samples from envs
        self.envs_till_idx = len(self._episodes)

    def _sample_episode(self, env_idx=None):
        idx = random.randint(0, self.envs_till_idx - 1) if env_idx is None else env_idx

        # sample idx with probability
        idx = np.random.choice(list(self._episodes.keys()))

        episode = random.choice(self._episodes[idx])
        return (episode, idx) if env_idx is None else episode

    def _sample(self):
        episodes, env_idx = self._sample_episode()
        observations = episodes["observation"]
        actions = episodes["action"]

        if self._obs_type == "pixels":
            # Sample obs, action
            sample_idx = np.random.randint(
                0, len(observations[self._pixel_keys[0]]) - self._history_len
            )
            sampled_pixel = {}
            for key in self._pixel_keys:
                sampled_pixel[key] = observations[key][
                    sample_idx : sample_idx + self._history_len
                ]
                sampled_pixel[key] = torch.stack(
                    [
                        self.aug(sampled_pixel[key][i])
                        for i in range(len(sampled_pixel[key]))
                    ]
                )

            if self._temporal_agg:
                # arrange sampled action to be of shape (history_len, num_queries, action_dim)
                sampled_action = np.zeros(
                    (self._history_len, self._num_queries, actions.shape[-1])
                )
                num_actions = (
                    self._history_len + self._num_queries - 1
                )  # -1 since its num_queries including the last action of the history
                act = np.zeros((num_actions, actions.shape[-1]))
                act[
                    : min(len(actions), sample_idx + num_actions) - sample_idx
                ] = actions[sample_idx : sample_idx + num_actions]
                if len(actions) < sample_idx + num_actions:
                    act[len(actions) - sample_idx :] = actions[-1]
                sampled_action = np.lib.stride_tricks.sliding_window_view(
                    act, (self._num_queries, actions.shape[-1])
                )
                sampled_action = sampled_action[:, 0]
            else:
                sampled_action = actions[sample_idx : sample_idx + self._history_len]

            return_dict = {}
            for key in self._pixel_keys:
                return_dict[key] = sampled_pixel[key]
            return_dict["actions"] = self.preprocess["actions"](sampled_action)

            # prompt
            if self._prompt == "text":
                return return_dict

        elif self._obs_type == "features":
            raise NotImplementedError

    def sample_test(self, env_idx, step=None):
        episode = self._sample_episode(env_idx)
        observations = episode["observation"]

        return_dict = {}

        if self._obs_type == "pixels":
            # observation
            if self._prompt == "text":
                for key in self._pixel_keys:
                    return_dict["prompt_" + key] = None
                return_dict["prompt_" + "proprioceptive"] = None
                return_dict["prompt_actions"] = None
            elif self._prompt == "goal":
                for key in self._pixel_keys:
                    prompt_pixel = np.transpose(observations[key][-1:], (0, 3, 1, 2))
                    return_dict["prompt_" + key] = prompt_pixel
                prompt_proprioceptive_state = np.concatenate(
                    [
                        observations["cartesian_states"][-1:],
                        observations["gripper_states"][-1:][:, None],
                    ],
                    axis=1,
                )
                return_dict["prompt_proprioceptive"] = self.preprocess[
                    "proprioceptive"
                ](prompt_proprioceptive_state)
                return_dict["prompt_actions"] = None
            elif self._prompt == "intermediate_goal":
                goal_idx = min(
                    step + self._intermediate_goal_step,
                    len(observations[self._pixel_keys[0]]) - 1,
                )
                for key in self._pixel_keys:
                    prompt_pixel = np.transpose(
                        observations[key][goal_idx : goal_idx + 1], (0, 3, 1, 2)
                    )
                    return_dict["prompt_" + key] = prompt_pixel
                prompt_proprioceptive_state = np.concatenate(
                    [
                        observations["cartesian_states"][goal_idx : goal_idx + 1],
                        observations["gripper_states"][goal_idx : goal_idx + 1][
                            :, None
                        ],
                    ],
                    axis=1,
                )
                return_dict["prompt_proprioceptive"] = self.preprocess[
                    "proprioceptive"
                ](prompt_proprioceptive_state)
                return_dict["prompt_actions"] = None

            return return_dict

        elif self._obs_type == "features":
            raise NotImplementedError

    def __iter__(self):
        while True:
            yield self._sample()

    def __len__(self):
        return self._num_samples

class ReplayBuffer:
    def __init__(self,
                 obs_shape,
                 action_dim,
                 batch_size=64,
                 max_size=int(1e3),
                 expert_size=int(1e5),
                 expert_ratio=1
                 ):
        self.max_size = max_size
        self.expert_max_size = expert_size
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
                    #self.save_expert_npz(f'all_retrieve_traj.npz')
                    print('=================Save expert done=========================')
                    add = False
                else:
                    self.add_expert_episode(self.current_ep_start_index, self.ptr)
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
            self.exp_obs['pixels'][exp_i] = traj['observations']['pixels'][idx].transpose(2, 0, 1)

            # copy actions
            self.exp_actions['policy'][exp_i] = traj['actions'][idx]

            # advance expert pointer
            self.expert_ptr = (self.expert_ptr + 1) % self.expert_max_size
            self.expert_size = min(self.expert_size + 1, self.expert_max_size)
            self.agu_start_ptr = self.expert_ptr



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
