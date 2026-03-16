import numpy as np
import os
import cv2
from retriever import get_retriever
import hydra
import torch
from regil import RegAgent


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


import numpy as np
file_path = '/home/carol/Project/4-RegIC_IL/ReG_IL_real/data/episode1.npy'
demo = np.load(file_path,allow_pickle=True)
print(f"Successfully loaded: {file_path} with length{len(demo)}")
