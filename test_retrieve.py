import numpy as np
import os
import cv2
from agent.retriever import get_retriever
import hydra
import torch
from franky import *
from env.franka_env import Franka,RobotEnv
import time
from collections import deque

def test_act_freq(path='/home/carolzhang/Project/RegIL/ReG_IL_real/expert_demos/reach.npy'):
    robot = Franka()
    robot.robot_reset()
    demo = np.load(path,allow_pickle=True).item()
    action = demo['actions']
    print(f'The demo is {len(action)}')
    act_freq = 1
    path_len = int(len(action))
    for i in range(path_len):
        robot.robot_act(action[i])
        time.sleep(0.1)
    print('finish the repaly')

def test_retriever(path= '/home/carolzhang/Project/RegIL/ReG_IL_real/expert_demos/reach.npy'):
    demo = np.load(path,allow_pickle=True).item()
    re_history_len = 5
    retiever = get_retriever(
        retrieve_key='DINO',
        state_num=5,
        metric='l2',
        traj_metric='sdtw',  # 'ot','sdtw'
        re_history_len=5,
        retrieve_len=5,
    )
    retiever.init_expert(demo)
    print("=" * 40)
    print(f"Load demo from: {path.split('/')[-1]}")
    print("=" * 40)
    env = RobotEnv()
    obs_que = deque(maxlen=re_history_len)
    obs,done = env.reset()
    obs_que.append(obs)
    while True:
        try:
            retrieved_act = get_retrieve_act(obs_que,retiever)
            next_obs,done = env.step(retrieved_act)
            obs_que.append(next_obs)
        except ControlException as e:
            print(f"Button Pressed, error detected: {e}")
            done = True
            while done:
                done,_ = env.get_done()
                time.sleep(0.2)

            success_input = input("Success or not(Y/N):")
            success = (success_input.upper() == "Y")
            print(f'This episode ended with {success},robot start reset')
            obs_que = deque(maxlen=re_history_len)
            time.sleep(0.5)
            obs, done = env.reset()
            obs_que.append(obs)



def get_retrieve_act(ob_history,retiever):
    current_traj = []
    for state_img in ob_history:
        current_traj.append(retiever.state_encode(state_img["pixels"]))

    state_subset = retiever.get_state_subset_from_task(current_traj)
    retrieve_state_idx_s, retrieve_state_idx_end, best_dist, path_len = retiever.get_traj_index_from_subset_traj(
        current_traj,
        state_subset,
    )
    end_idx = min(retrieve_state_idx_end + retiever.retrieve_len, len(retiever.exp_traj['actions']))
    retrieved_act = retiever.exp_traj['actions'][retrieve_state_idx_end:end_idx][0]

    return retrieved_act

@hydra.main(config_path="cfgs", config_name="config")
def main(cfg):
    from train_robot import WorkspaceIL as W
    from train_robot import make_agent
    #1. test action frequency
    #test_act_freq()


    #2. test retrieve
    #test_retriever()

    #3.test work space
    cfg.save_video = True
    workspce= W(cfg)

    task_idx = 0
    data_path = cfg.suite.data_path
    #task = cfg.suite.task.tasks[task_idx]
    #raw_act_stat, max_episode_len,all_demo = workspce.preprocess_demo(data_path, task)

    # create envs
    #cfg.suite.task_make_fn.max_episode_len = 300
    #cfg.suite.task_make_fn.act_max = cfg.suite.task_make_fn.act_max
    #cfg.suite.task_make_fn.act_min = raw_act_stat['min'].tolist()
    env = workspce.env
    agent = workspce.agent

    agent.init_demos(all_demo, skip=cfg.suite.demo_skip)
    observation, done = env.reset()
    agent.buffer_reset(observation)
    with torch.no_grad():
        policy_action = agent.act(
                                  obs=observation.copy(),
                                  retrieve_only=True
                                       )
        next_obs, done = env.step(policy_action)
        agent.update_obs_and_retrieve(next_obs)
        retrive_reward, retrieve_action, reward_dict = agent.get_reward(next_obs)
        agent.add_buffer(observation, next_obs, done, policy_action, retrive_reward, retrieve_action)


main()