#!/usr/bin/env python3

import warnings
import time
from pathlib import Path
from franky import *
import hydra
import torch
import random
import numpy as np
from pathlib import Path

import utils.net_utils as utils
from logger import Logger
from video import VideoRecorder

warnings.filterwarnings("ignore", category=DeprecationWarning)
torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg):
    obs_shape = {}
    for key in cfg.suite.pixel_keys:
        obs_shape[key] = obs_spec.shape
    cfg.agent.obs_shape = obs_shape
    cfg.agent.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg.agent)

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

class WorkspaceIL:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg
        set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None
        )

        # load data
        task_idx = 0
        data_path =self.cfg.suite.data_path
        task = self.cfg.suite.task.tasks[task_idx]
        raw_act_stat,max_episode_len,_ = self.preprocess_demo(data_path,task)

        if self.cfg.agent.name == 'bc':
            self.cfg.suite.num_train_steps = 100000
            self.cfg.suite.save_every_steps = self.cfg.suite.eval_every_steps
            self.cfg.suite.task_make_fn.use_robot = False


        # create envs
        self.cfg.suite.task_make_fn.max_episode_len = 200
        self.cfg.suite.task_make_fn.act_max = raw_act_stat['max'].tolist()
        self.cfg.suite.task_make_fn.act_min = raw_act_stat['min'].tolist()
        self.env = hydra.utils.call(self.cfg.suite.task_make_fn)

        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb,mode='rl')

        # create agent
        self.agent = make_agent(
            self.env.observation_spec, self.env.action_spec, cfg
        )




        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 1
        self.episode_step = 0



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
        self.all_demo = {
            'observations': data['observations'],
            'actions': action_processed,
        }

        return act_stat,max_episode_len,self.all_demo



    def eval(self):
        
        self.agent.train(False)
        success_list,episode_rewards = [],[]
        episode = 0
        ep_reward_dict = {}
        self.agent.init_demos(self.all_demo,skip=self.cfg.suite.demo_skip)
        eval_until_episode = utils.Until(self.cfg.suite.num_eval_episodes)
        while eval_until_episode(episode):
            print(f'======================Start eval EP{episode+1}/{self.cfg.suite.num_eval_episodes}==============================')
            
            observation,done = self.env.reset()

            self.agent.buffer_reset(observation)
            step = 0
            total_reward = 0
            self.video_recorder.init(observation['pixels'], enabled=True)

            # plot obs with cv2
            while not done :
                try:

                    with torch.no_grad(), utils.eval_mode(self.agent):
                        action = self.agent.act(
                            observation,
                            eval_mode=True,
                        )

                    next_observation, done = self.env.step(action.squeeze())
                    time.sleep(0.1)
                    self.video_recorder.record(next_observation['pixels'])
                    self.agent.update_obs_and_retrieve(next_observation)
                    retrive_reward, retrieve_action,reward_dict = self.agent.get_reward()
                    total_reward = total_reward + retrive_reward
                    for name in reward_dict.keys():
                        ep_reward_dict[name] = ep_reward_dict.get(name, 0) + reward_dict[name]
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
            episode_rewards.append(total_reward / episode)
            success_list.append(success)

        with self.logger.log_and_dump_ctx(self.global_step, ty="eval") as log:
            for env_idx, reward in enumerate(episode_rewards):
                log(f"episode_reward", reward)
            log("episode", episode)
            log("success_rate", np.mean(success_list))
            log("step", self.global_step)
            for name in ep_reward_dict.keys():
                log(name, ep_reward_dict[name] / (episode + 1e-6))

        self.agent.train(True)

    def train(self):
        ep_reward_list = []
        success_list = []
        episode_step = 0

        # Initialize demonstrations for this environment
        self.agent.init_demos(self.all_demo,skip=self.cfg.suite.demo_skip)

        # Step schedulers
        train_until_step = utils.Until(self.cfg.suite.num_train_steps, 1)
        log_every_step = utils.Every(self.cfg.suite.log_every_steps, 1)
        eval_every_step = utils.Every(self.cfg.suite.eval_every_steps, 1)
        save_every_step = utils.Every(self.cfg.suite.save_every_steps, 1)


        observation,done = self.env.reset()
        self.agent.buffer_reset(observation)
        episode_reward = 0
        ep_reward_dict = {}

        while train_until_step(self.global_step):
            try:

                if (
                    self.cfg.eval
                    and eval_every_step(self.global_step)
                    and self.global_step > 0
                ):
                    self.logger.log(
                        "eval_total_time", self.timer.total_time(), self.global_frame
                    )
                    self.eval()
                    observation,done = self.env.reset()
                    self.agent.buffer_reset(observation)
                    episode_step = 0
                with torch.no_grad():
                    policy_action = self.agent.act(
                        obs = observation.copy(),
                        #retrieve_only=True
                                            )

                next_observation,done = self.env.step(policy_action)
                episode_step = episode_step  +1

                self.agent.update_obs_and_retrieve(next_observation)

                retrive_reward,retrieve_action,reward_dict = self.agent.get_reward()
                episode_reward += retrive_reward
                for name in reward_dict.keys():
                    ep_reward_dict[name] = ep_reward_dict.get(name,0) + reward_dict[name]
                success = False
                if done:
                    success_input = input(f"Achive the max length, Success or not(Y/N):")
                    success = (success_input.upper() == "Y")
                    success_list.append(success)

                self.agent.add_buffer(observation, next_observation,done,policy_action,retrive_reward, retrieve_action,success=success)
                metrics = self.agent.update()
                self.logger.log_metrics(metrics, self.global_frame, ty="train")
                if self.global_step >1000:
                    for i in range(3):
                        metrics = self.agent.update()
                        self.logger.log_metrics(metrics, self.global_frame, ty="train")


                if done :
                    ep_reward_list.append(episode_reward)
                    print(
                        f'EP{self.global_episode}: ended with {success}, Step {episode_step},Episode_reward {episode_reward}')
                    # reset episode
                    observation,done = self.env.reset()
                    self.agent.buffer_reset(observation)
                    self._global_episode = self._global_episode + 1
                    episode_step = 0
                    episode_reward = 0

                else:
                    observation = next_observation

                # log
                if log_every_step(self.global_step+1):
                    elapsed_time, total_time = self.timer.reset()
                    with self.logger.log_and_dump_ctx(self.global_frame, ty="train") as log:
                        log("total_time", total_time)
                        log("step", self.global_step)
                        log("episode", np.array(len(success_list)))
                        log("reward", np.mean(ep_reward_list) )
                        log("success_rate", np.mean(success_list))
                        for name in metrics.keys():
                            log(name, metrics[name])

                        for name in ep_reward_dict.keys():
                            log(name, ep_reward_dict[name]/( len(ep_reward_list)+1e-6))
                        ep_reward_list = []
                        ep_reward_dict = {}
                        success_list = []


                # save snapshot
                if save_every_step(self.global_step):
                    self.save_snapshot()
                self._global_step += 1

            except ControlException as e:
                print(f"Button Pressed, error detected: {e}")
                is_button = True

                while is_button:
                    is_button, _ = self.env.get_done()
                    time.sleep(0.5)

                success_input = input(f"Success or not(Y/N):")
                success = (success_input.upper() == "Y")
                success_list.append(success)
                observation = next_observation
                next_observation = self.env.get_observation()
                
                self.agent.update_obs_and_retrieve(next_observation)
                retrive_reward, retrieve_action, reward_dict = self.agent.get_reward()
                episode_reward = episode_reward + retrive_reward
                ep_reward_list.append(episode_reward)
                done = True
                self.agent.add_buffer(observation, next_observation, done, self.env.input_action, retrive_reward,
                                      retrieve_action,success=success)
                print(f'EP {self.global_episode}: ended with {success}, Step {episode_step},Episode_reward {episode_reward}')
                #print("-" * 40)

                
                observation, done = self.env.reset()
                time.sleep(2)
                episode_step = 0

                self.agent.buffer_reset(observation)
                episode_reward = 0
                self._global_episode += 1


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


@hydra.main(config_path="cfgs", config_name="config")
def main(cfg):
    #from train_robot import WorkspaceIL as W

    root_dir = Path.cwd()
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
        workspace.eval()

    workspace.train()


if __name__ == "__main__":
    main()

# libfranka: Move command aborted: motion aborted by reflex! ["cartesian_motion_generator_velocity_discontinuity", "cartesian_motion_generator_acceleration_discontinuity", "cartesian_motion_generator_joint_acceleration_discontinuity"]