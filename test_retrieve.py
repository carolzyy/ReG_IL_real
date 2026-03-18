import numpy as np
import os
import cv2
from agent.retriever import get_retriever
import hydra
import torch
from franky import *
from env.franka_env import Franka
import time

re_history_len = 5
retiever = get_retriever(
    retrieve_key='DINO',
    state_num=5,
    metric='l2',
    traj_metric='sdtw',  # 'ot','sdtw'
    re_history_len=5,
    retrieve_len=5,
)

robot = Franka()
robot.robot_reset()
path = '/home/carolzhang/Project/RegIL/ReG_IL_real/expert_demos/reach.npy'
demo = np.load(path,allow_pickle=True).item()
action = demo['actions']
for i in range(len(action)):
    robot.robot_act(action[i])
    #print(f'excute the {i}th action')
    #time.sleep(0.05)


retiever.init_expert(demo)


obs = np.load('/home/carolzhang/Project/RegIL/ReG_IL_real/recorded/episode0.npy',allow_pickle=True)
for i in range(0,128):
    state_idx = i
    #print(f'state_idx is {i}')
    start_idx_start = max(state_idx - re_history_len, 0)
    ob_history = obs[start_idx_start: state_idx+1] # dict{'dino','clip'}
    current_traj = []
    for state_img in ob_history:
        current_traj.append(retiever.state_encode(state_img['image']))

    state_subset = retiever.get_state_subset_from_task(current_traj,demo)
    retrieve_state_idx_s,retrieve_state_idx_end,best_dist,path_len = retiever.get_traj_index_from_subset_traj(
                        current_traj,
                        state_subset,
                        #retiever.exp_traj
    )
    end_idx = min(retrieve_state_idx_end + retiever.retrieve_len, len(retiever.exp_traj['actions']))
    retrieved_act = retiever.exp_traj['actions'][retrieve_state_idx_end:end_idx][0]

    print(f'Demo {i}th state_idx : retrieve_idx {retrieve_state_idx_end}, action is {retrieved_act}')
