#!/usr/bin/env python3

import warnings
import hydra
import torch
import random
import numpy as np
from pathlib import Path
from franky import *
import utils.net_utils as utils
from logger import Logger
from video import VideoRecorder
import time
warnings.filterwarnings("ignore", category=DeprecationWarning)
torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg):
    obs_shape = {}
    for key in cfg.suite.pixel_keys:
        obs_shape[key] = obs_spec.shape
    cfg.agent.obs_shape = obs_shape
    print(f'The agent obs shape is {obs_shape}')
    cfg.agent.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg.agent)


class WorkspaceIL:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None
        )

        # load data
        task_idx = 0
        data_path =self.cfg.suite.data_path
        task = self.cfg.suite.task.tasks[task_idx]
        max_episode_len,buff = self.preprocess_buff(task)
        raw_act_stat,_,demo = self.preprocess_demo(data_path,task)
        self.all_demo = buff

        self.cfg.suite.num_train_steps = 10000
        self.cfg.suite.save_every_steps = self.cfg.suite.eval_every_steps
        self.cfg.suite.task_make_fn.use_robot = False

        # create envs
        self.cfg.suite.task_make_fn.max_episode_len = 250
        self.cfg.suite.task_make_fn.act_max = raw_act_stat['max'].tolist()
        self.cfg.suite.task_make_fn.act_min = raw_act_stat['min'].tolist()
        self.env = hydra.utils.call(self.cfg.suite.task_make_fn)

        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb,mode='bc')

        # create agent
        self.agent = make_agent(
            self.env.observation_spec, self.env.action_spec, cfg
        )

        self.timer = utils.Timer()
        self._global_step = 0



    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.suite.action_repeat

    def preprocess_demo(self,data_path,task):
        data = np.load(f'{data_path}/{task}.npy',allow_pickle=True).item()

        action = data['actions']
        obs = data['observations']['pixels']
        self.video_recorder.init(obs[0])
        for image in obs[1:]:
            self.video_recorder.record(image)
        self.video_recorder.save('demo.mp4')


        max_episode_len = len(action)
        print(f'Load demo from {data_path}/{task}.npy, \n'
              f'Demo length : {max_episode_len},\n'
              f'Obs shape: {obs.shape}'
              )

        act_stat = {
            'max': action.max(axis=0),
            'min': action.min(axis=0),
        }
        act_range = act_stat['max'][:3] - act_stat['min'][:3]
        action_xyz = 2*(action[..., :3]-act_stat['min'][:3])/(act_range+ 1e-8)-1
        action_gripper = np.where(action[..., 3:] > 0.5, 1.0, -1.0)
        action_processed = np.concatenate([action_xyz, action_gripper], axis=-1)
        all_demo = {
            'observations': data['observations'],
            'actions': action_processed,
        }

        return act_stat,max_episode_len,all_demo

    def preprocess_buff(self,task):
        data_path = f'/home/carol/Project/4-RegIC_IL/ReG_IL_real/expert_demos/retrieve_buff/{task}_expert_buffer.npz'
        data = np.load(data_path, allow_pickle=True)
        action = data['action_policy']
        obs = data['obs_pixels']
        for image in obs[1:]:
            self.video_recorder.record(image)
        self.video_recorder.save('demo.mp4')

        max_episode_len = len(action)
        print(f'Load demo from {data_path}, \n'
              f'Demo length : {max_episode_len},\n'
              f'Obs shape: {obs.shape}'
              )

        all_demo = {
            'observations': {'pixels':obs},
            'actions': action,
        }

        return max_episode_len, all_demo


    def train(self):
        # Initialize demonstrations for this environment
        self.agent.buff.add_demo(self.all_demo)

        # Step schedulers
        train_until_step = utils.Until(self.cfg.suite.num_train_steps, 1)
        log_every_step = utils.Every(self.cfg.suite.log_every_steps, 1)
        save_every_step = utils.Every(self.cfg.suite.save_every_steps, 1)

        while train_until_step(self.global_step):
            metrics = self.agent.update()
            self.logger.log_metrics(metrics, self.global_frame, ty="train")
            # log
            if log_every_step(self.global_step+1):
                elapsed_time, total_time = self.timer.reset()
                with self.logger.log_and_dump_ctx(self.global_frame, ty="train") as log:
                    log("total_time", total_time)
                    log("step", self.global_step)
                    for name in metrics.keys():
                        log(name, metrics[name])

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
        self.agent.load_snapshot(agent_payload, eval=False)

    def eval(self):
        self.agent.train(False)
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.suite.num_eval_episodes)
        success_list = []

        while eval_until_episode(episode):
            print(
                f'======================Start eval EP{episode + 1}/{self.cfg.suite.num_eval_episodes}==============================')
            observation,done = self.env.reset()
            self.video_recorder.init(observation['pixels'], enabled=True)

            while not done:
                try:
                    with torch.no_grad(), utils.eval_mode(self.agent):
                        # select_action usually takes obs and returns a numpy array
                        action = self.agent.act(observation,
                                                eval_mode=True,)

                        next_observation, done = self.env.step(action.squeeze())
                        time.sleep(0.1)
                        self.video_recorder.record(next_observation['pixels'])
                        step += 1
                        observation = next_observation
                except ControlException as e:
                    print(f"Button Pressed, error detected: {e}")
                    is_button = True

                    while is_button:
                        is_button, _ = self.env.get_done()
                        time.sleep(0.2)

                    time.sleep(0.5)
                    break
            success_input = input(f"Success or not(Y/N):")
            success = (success_input.upper() == "Y")
            print(
                f'This episode ended with {success}, eposide{episode},Step{step},robot start reset')
            print("-" * 40)

            episode += 1
            self.video_recorder.save(f"eval_{self.global_step}_{episode}.mp4")
            success_list.append(success)

        with self.logger.log_and_dump_ctx(self.global_frame, ty="eval") as log:
            log("episode", episode)
            log("success_rate", np.mean(success_list))
            log("step", self.global_step)


@hydra.main(config_path="cfgs", config_name="config")
def main(cfg):
    workspace = WorkspaceIL(cfg)

    # Load weights
    if cfg.eval:
        snapshots = {}
        bc_snapshot = Path(cfg.bc_weight)
        if not bc_snapshot.exists():
            raise FileNotFoundError(f"bc weight not found: {bc_snapshot}")
        print(f"loading bc weight: {bc_snapshot}")
        snapshots["bc"] = bc_snapshot
        workspace.load_snapshot(snapshots)
        workspace.train()


    workspace.train()


if __name__ == "__main__":
    main()