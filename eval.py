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
from datetime import datetime
import utils.net_utils as utils
from logger import Logger
from video import VideoRecorder
from omegaconf import OmegaConf
warnings.filterwarnings("ignore", category=DeprecationWarning)
torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg):
    obs_shape = {}
    for key in cfg.suite.pixel_keys:
        obs_shape[key] = obs_spec.shape
    cfg.agent.obs_shape = obs_shape
    cfg.agent.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg.agent)

def make_env(cfg,work_dir):
    task_idx = 0
    data_path = cfg.suite.data_path
    task = cfg.suite.task.tasks[task_idx]
    raw_act_stat, max_episode_len, _ = preprocess_demo(work_dir,data_path, task,)

    cfg.suite.task_make_fn.use_robot = False

    # create envs
    max_episode_len = 250
    cfg.suite.task_make_fn.max_episode_len = max_episode_len

    cfg.suite.task_make_fn.act_max = raw_act_stat['max'].tolist()
    cfg.suite.task_make_fn.act_min = raw_act_stat['min'].tolist()
    return hydra.utils.call(cfg.suite.task_make_fn),max_episode_len


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def preprocess_demo(work_dir,data_path, task):
    video_recorder = VideoRecorder(
            work_dir
        )
    data = np.load(f'{data_path}/{task}.npy', allow_pickle=True).item()

    action = data['actions']
    obs = data['observations']['pixels']
    video_recorder.init(obs[0])
    for image in obs[1:]:
        video_recorder.record(image)
    video_recorder.save(f'demo_{task}.mp4')

    max_episode_len = len(action)
    print(f'Load demo from {data_path}/{task}.npy, \n'
          f'Demo length : {max_episode_len},\n'
          )

    act_stat = {
        'max': action.max(axis=0),
        'min': action.min(axis=0),
    }
    act_range = act_stat['max'][:3] - act_stat['min'][:3]
    action_xyz = 2 * (action[..., :3] - act_stat['min'][:3]) / (act_range + 1e-8) - 1
    action_gripper = np.where(action[..., 3:] > 0.5, 1.0, -1.0)
    action_processed = np.concatenate([action_xyz, action_gripper], axis=-1)
    all_demo = {
        'observations': data['observations'],
        'actions': action_processed,
    }

    return act_stat, max_episode_len, all_demo

class WorkspaceIL:
    def __init__(self,
                 cfg,
                 work_dir,
                 env):


        self.work_dir = work_dir/cfg.agent.name
        print(f"workspace: {self.work_dir}")
        self.env = env

        self.cfg = cfg
        set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None
        )



        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb, mode='rl')

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



    def eval(self,round):
        self.agent.train(False)
        episode = 0
        success_list = []
        eval_until_episode = utils.Until(self.cfg.suite.num_eval_episodes) #
        while eval_until_episode(episode):
            print(
                f'======================Start eval SCE{round}-EP{episode + 1}/{self.cfg.suite.num_eval_episodes}==============================')

            observation, done = self.env.reset()

            #self.agent.buffer_reset(observation)
            step = 0
            total_reward = 0
            self.video_recorder.init(observation['pixels'], enabled=True)

            # plot obs with cv2
            while not done:
                try:

                    with torch.no_grad(), utils.eval_mode(self.agent):
                        action = self.agent.act(
                            observation,
                            eval_mode=True,
                            step = step,
                            global_step = step,
                        )

                    next_observation, done = self.env.step(action.squeeze())
                    time.sleep(0.1)
                    self.video_recorder.record(next_observation['pixels'])
                    #self.agent.update_obs_and_retrieve(next_observation)
                    #retrive_reward, retrieve_action, reward_dict = self.agent.get_reward()
                    #total_reward = total_reward + retrive_reward
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
            success_list.append(success)

            print(
                f'This episode ended with {success}, eposide{episode},Step{step},robot start reset')
            print("-" * 40)

            episode += 1
            self.video_recorder.save(f"eval_{round}_{episode}.mp4")

        with self.logger.log_and_dump_ctx(self.global_step, ty="eval") as log:
            log("episode", episode)
            log("success_rate", np.mean(success_list))
            log("step", self.global_step)


    def load_snapshot(self, snapshots):
        # bc
        with snapshots["bc"].open("rb") as f:
            payload = torch.load(f,weights_only=False)
        agent_payload = {}
        for k, v in payload.items():
            if k not in self.__dict__:
                agent_payload[k] = v
        self.agent.load_snapshot(agent_payload, eval=False)


#@hydra.main(config_path="cfgs", config_name="config")
def main():
    eval_list = [
        {"name": "BC",
         "config": "/media/carol/KINGSTON/RegIL/model/03.25_train/bc/144140/.hydra/config.yaml"},
        {"name": "BAKU",
         "config": "/media/carol/KINGSTON/RegIL/model/03.25_train/baku/145857/.hydra/config.yaml"},
        {"name": "ReGIL",
         "config": "/media/carol/KINGSTON/RegIL/model/regil/211951/.hydra/config.yaml"},
    ]
    timestamp = datetime.now().strftime("%m-%d")
    work_dir = Path.cwd() / f"exp_local/{timestamp}_eval"
    env = None
    scene_num = 5

    for scene_id in range(scene_num):
        for item in eval_list:
            print(f"\n" + "=" * 30+f"EVALUATING: {item['name']}"+ "=" * 30)
            cfg = OmegaConf.load(item['config'])
            cfg.expert_dataset = item['name']
            OmegaConf.resolve(cfg)
            if env is None:
                env,max_len = make_env(cfg,work_dir)
            cfg.agent.max_episode_len = max_len
            workspace = WorkspaceIL(cfg=cfg,env=env,work_dir=work_dir)
            snapshots = {}
            model_path = item['config'].split('.hydra')[0]+f'snapshot/snapshot.pt'
            bc_snapshot = Path(model_path)
            if not bc_snapshot.exists():
                raise FileNotFoundError(f"bc weight not found: {bc_snapshot}")
            print(f"loading bc weight: {bc_snapshot}")
            snapshots["bc"] = bc_snapshot
            workspace.load_snapshot(snapshots)
            workspace.eval(scene_id)



if __name__ == "__main__":
    main()