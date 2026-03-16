#!/usr/bin/env python3

import warnings
import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path

import hydra
import torch
import cv2
import numpy as np
from pathlib import Path

import utils
from logger import Logger
from replay_buffer import make_expert_replay_loader
from video import VideoRecorder, TrainVideoRecorder

warnings.filterwarnings("ignore", category=DeprecationWarning)
torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg):
    obs_shape = {}
    for key in cfg.suite.pixel_keys:
        obs_shape[key] = obs_spec[key].shape
    if cfg.use_proprio:
        obs_shape[cfg.suite.proprio_key] = obs_spec[cfg.suite.proprio_key].shape
    obs_shape[cfg.suite.feature_key] = obs_spec[cfg.suite.feature_key].shape
    cfg.agent.obs_shape = obs_shape
    cfg.agent.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg.agent)

def sample_bounded_array(bounded_spec):
    low = bounded_spec.minimum
    high = bounded_spec.maximum
    shape = bounded_spec.shape
    return np.random.uniform(low, high, size=shape).astype(bounded_spec.dtype)


class WorkspaceIL:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # load data
        dataset_iterable = hydra.utils.call(self.cfg.expert_dataset)
        self.expert_replay_loader = make_expert_replay_loader(
            dataset_iterable, self.cfg.agent.batch_size
        )
        self.expert_replay_iter = iter(self.expert_replay_loader)
        self.stats = self.expert_replay_loader.dataset.stats

        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb,mode='rl')
        # create envs
        self.cfg.suite.task_make_fn.max_episode_len = (
            self.expert_replay_loader.dataset._max_episode_len
        )
        self.cfg.suite.task_make_fn.max_state_dim = (
            self.expert_replay_loader.dataset._max_state_dim
        )
        if self.cfg.suite.name == "dmc":
            self.cfg.suite.task_make_fn.max_action_dim = (
                self.expert_replay_loader.dataset._max_action_dim
            )

        self.env, self.task_descriptions = hydra.utils.call(self.cfg.suite.task_make_fn)

        # create agent
        self.agent = make_agent(
            self.env[0].observation_spec(), self.env[0].action_spec(), cfg
        )
        if getattr(self.agent, 'reward_type', False) == 'tot':
            self.env[0]._env._env._env._stop_early = False
            if getattr(self.env[0]._env._env._env._env, 'max_path_length', False):
                self.env[0]._env._env._env._env.max_path_length = self.expert_replay_loader.dataset._max_episode_len

        if self.cfg.agent.name == 'bc':
            self.cfg.suite.num_train_steps = 100000
            self.cfg.suite.save_every_steps = self.cfg.suite.eval_every_steps

        self.all_demos = dataset_iterable._episodes

        self.envs_till_idx = self.expert_replay_loader.dataset.envs_till_idx

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 1

        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None
        )
        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None
        )

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.suite.action_repeat

    def eval(self):
        self.agent.train(False)
        episode_rewards_gt,episode_rewards = [],[]
        successes = []
        ep_reward_dict = {}

        num_envs = self.envs_till_idx


        for env_idx in range(num_envs):
            print(f"evaluating env {env_idx}")
            episode, total_reward_gt,total_reward = 0, 0, 0
            eval_until_episode = utils.Until(self.cfg.suite.num_eval_episodes)
            success = []

            while eval_until_episode(episode):
                time_step = self.env[env_idx].reset()
                self.agent.buffer_reset(time_step.observation)
                step = 0


                if episode == 0:
                    self.video_recorder.init(self.env[env_idx], enabled=True)

                # plot obs with cv2
                while not time_step.last() :

                    with torch.no_grad(), utils.eval_mode(self.agent):
                        action = self.agent.act(
                            time_step.observation,
                            step=step,
                            eval_mode=True,
                        )
                    time_step = self.env[env_idx].step(action.squeeze())
                    self.video_recorder.record(self.env[env_idx])
                    retrive_reward, retrieve_action,reward_dict = self.agent.get_reward(time_step)
                    if retrive_reward is not None:
                        total_reward = total_reward + retrive_reward
                    total_reward_gt += time_step.reward
                    for name in reward_dict.keys():
                        ep_reward_dict[name] = ep_reward_dict.get(name, 0) + reward_dict[name]
                    step += 1

                episode += 1
                success.append(time_step.observation["goal_achieved"])
            self.video_recorder.save(f"{self.global_step}_env{env_idx}.mp4")
            episode_rewards_gt.append(total_reward_gt / episode)
            episode_rewards.append(total_reward / episode)
            successes.append(np.mean(success))

        for _ in range(len(self.env) - num_envs):
            episode_rewards.append(0)
            successes.append(0)

        with self.logger.log_and_dump_ctx(self.global_step, ty="eval") as log:
            for env_idx, reward in enumerate(episode_rewards):
                log(f"episode_reward_env{env_idx}", reward)
                log(f"success_env{env_idx}", successes[env_idx])
            log("episode_reward", np.mean(episode_rewards[:num_envs]))
            log("episode_reward_gt", np.mean(episode_rewards_gt[:num_envs]))
            log("success_rate", np.mean(successes))
            log("episode_length", step * self.cfg.suite.action_repeat )
            log("episode", episode)
            log("step", self.global_step)
            for name in ep_reward_dict.keys():
                log(name, ep_reward_dict[name] / (episode + 1e-6))

        self.agent.train(True)

    def train(self):
        ep_mean_reward_list = []
        ep_mean_gt_reward_list = []
        success_list =[]
        env_idx = 0
        pixels = []
        time_steps = []

        # Initialize demonstrations for this environment
        self.agent.init_demos(self.all_demos[env_idx],skip=self.cfg.suite.demo_skip)

        # Step schedulers
        train_until_step = utils.Until(self.cfg.suite.num_train_steps, 1)
        #log_every_episodes = utils.Every( self.cfg.suite.log_every_episodes**100, 1) #self.cfg.suite.log_every_steps
        log_every_step = utils.Every(self.cfg.suite.log_every_episodes *200, 1)
        eval_every_step = utils.Every(self.cfg.suite.eval_every_steps, 1)
        save_every_step = utils.Every(self.cfg.suite.save_every_steps, 1)


        time_step = self.env[env_idx].reset()
        self.agent.buffer_reset(time_step.observation)

        episode_step = 0
        episode_reward = 0
        episode_reward_gt = 0
        ep_reward_dict = {}
        time_steps.append(time_step)


        while train_until_step(self.global_step):
            if (self.global_episode % self.cfg.video_every_episode == 0) and not (self.train_video_recorder.enabled):
                self.train_video_recorder.init(time_step.observation['pixels'], enabled=True)
            if (
                self.cfg.eval
                and eval_every_step(self.global_step)
                and self.global_step > 0
            ):
                self.logger.log(
                    "eval_total_time", self.timer.total_time(), self.global_frame
                )
                self.eval()
                pixels = []
                time_steps = []
                time_step = self.env[env_idx].reset()
                self.agent.buffer_reset(time_step.observation)
                time_steps.append(time_step)
            with torch.no_grad():
                #expl_noise =  0.1 #utils.schedule(f"linear(0.4,0, {self.cfg.suite.num_train_steps})", self.global_step)

                action = self.agent.act(obs = time_step.observation.copy(),
                                        step = episode_step,
                                        expl_noise = 0.1,
                                        )#.astype(np.float32)

            next_time_step = self.env[env_idx].step(action)

            self.agent.update_obs_and_retrieve(next_time_step.observation)

            if self.train_video_recorder.enabled:
                self.train_video_recorder.record(next_time_step.observation['pixels'])

            retrive_reward,retrieve_action,reward_dict = self.agent.get_reward(next_time_step)
            if retrive_reward is None:
                time_steps.append(next_time_step)
                pixels.append(next_time_step.observation['pixels'])
            else:
                episode_reward += retrive_reward
                episode_reward_gt += next_time_step.reward
                for name in reward_dict.keys():
                    ep_reward_dict[name] = ep_reward_dict.get(name,0) + reward_dict[name]
                self.agent.add_buffer(time_step, next_time_step,retrive_reward,retrieve_action)


            metrics = self.agent.update() #self.expert_replay_iter

            self.logger.log_metrics(metrics, self.global_frame, ty="train")


            if next_time_step.last() :
                if self.train_video_recorder.enabled:
                    self.train_video_recorder.save(
                        f"train_step{self.global_step}_ep{self._global_episode}.mp4"
                    )
                    self.train_video_recorder.enabled = False

                success = next_time_step.observation["goal_achieved"]
                ep_mean_reward_list.append(episode_reward)
                ep_mean_gt_reward_list.append(episode_reward_gt)
                success_list.append(success)
                if self.agent.reward_type == 'tot':
                    pixels = np.stack(pixels, axis=0)
                    ot_rewards, cost_min, cost_max = self.agent.tot_rewarder(pixels)
                    assert cost_min >= 0

                    if self.global_episode == 1:
                        ot_rewards_sum = abs(ot_rewards.sum())
                        print(f'ot_rewards_sum = {ot_rewards_sum}')
                        self.agent.sinkhorn_rew_scale = 10.0 / ot_rewards_sum #self.agent.auto_rew_scale_factor=10
                        ot_rewards, cost_min, cost_max = self.agent.tot_rewarder(pixels)

                    for i, processed_elt in enumerate(time_steps):
                        #processed_elt = elt._replace(observation=time_steps[i].observation['pixels'])
                        if i == 0:
                            current_elt = processed_elt
                            continue
                        reward=ot_rewards[i - 1]
                        self.agent.add_buffer(current_elt, processed_elt,reward)
                        current_elt = processed_elt
                    pixels = []
                    time_steps = []

                # reset episode
                time_step = self.env[env_idx].reset()
                self.agent.buffer_reset(time_step.observation)
                time_steps.append(time_step)



                should_record = (self._global_episode % self.cfg.video_every_episode == 0)
                if should_record and self.cfg.save_train_video:
                    # init will set enabled=True for the recorder; make sure it's done only once
                    self.train_video_recorder.init(time_step.observation['pixels'], enabled=True)

                episode_reward = 0
                episode_reward_gt = 0

                episode_step = 0
                self._global_episode = self._global_episode + 1

            else:
                time_step = next_time_step
                episode_step += 1

            # log
            if log_every_step(self.global_step+1):
                elapsed_time, total_time = self.timer.reset()
                with self.logger.log_and_dump_ctx(self.global_frame, ty="train") as log:
                    log("total_time", total_time)
                    log("step", self.global_step)
                    log("episode", np.array(len(success_list)))
                    log("success_rate", np.array(success_list).sum() )
                    log("reward", np.mean(ep_mean_reward_list) ) #.sum() / len(ep_mean_reward_list))
                    log("reward_gt", np.mean(ep_mean_gt_reward_list) ) #.sum() / len(ep_mean_gt_reward_list))
                    # print(f'Success is {np.array(success_list).sum()}/{len(success_list)}')
                    for name in metrics.keys():
                        log(name, metrics[name])

                    for name in ep_reward_dict.keys():
                        log(name, ep_reward_dict[name]/( len(ep_mean_reward_list)+1e-6))
                    success_list = []
                    ep_mean_reward_list = []
                    ep_reward_dict = {}
                    ep_mean_gt_reward_list = []


            # save snapshot
            if save_every_step(self.global_step):
                self.save_snapshot()
            self._global_step += 1




    def save_snapshot(self):
        snapshot_dir = self.work_dir / "snapshot"
        snapshot_dir.mkdir(exist_ok=True)
        self.agent.save_snapshot(snapshot_dir)

    def load_snapshot(self, snapshots):
        # bc
        with snapshots["bc"].open("rb") as f:
            payload = torch.load(f)
        agent_payload = {}
        for k, v in payload.items():
            if k not in self.__dict__:
                agent_payload[k] = v
        if "vqvae" in snapshots:
            with snapshots["vqvae"].open("rb") as f:
                payload = torch.load(f)
            agent_payload["vqvae"] = payload
        self.agent.load_snapshot(agent_payload, eval=False)


@hydra.main(config_path="cfgs", config_name="config")
def main(cfg):
    from train_rl import WorkspaceIL as W

    root_dir = Path.cwd()
    workspace = W(cfg)

    # Load weights
    if cfg.load_bc:
        snapshots = {}
        bc_snapshot = Path(cfg.bc_weight)
        if not bc_snapshot.exists():
            raise FileNotFoundError(f"bc weight not found: {bc_snapshot}")
        print(f"loading bc weight: {bc_snapshot}")
        snapshots["bc"] = bc_snapshot
        workspace.load_snapshot(snapshots)

    workspace.train()


if __name__ == "__main__":
    main()
