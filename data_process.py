import torch
import torch.nn.functional as F
import numpy as np
from utils.encoders import get_encoders
from pathlib import Path
from scipy.spatial.transform import Rotation

def data_process(path='',retrieve_key='DINO'):
    # 1. load the saved data
    demo = np.load(path,allow_pickle=True)
    print(f'The demo is {len(demo)} length, include info: {demo[0].keys()}')
    encoder = None
    if retrieve_key is not None:
        encoder = get_encoders()
        #encoder = encoder[retrieve_key]
        print(f'Process the image with encoder {encoder.keys()}')
    feature_traj = []
    act_traj = []
    pixel_traj = []


    for idx in range(len(demo)-1):
        image = demo[idx]['image'] # resize to 128,128 for the buffer size
        pixel_traj.append(image)
        retrieve_feature = {}
        if encoder is not None:
            for name in encoder.keys():
                retrieve_feature[name] = encoder[name].encode(image.copy())
            feature_traj.append(retrieve_feature)
        gripper = demo[idx]['gripper_command']
        delta_p = get_action_matrix(demo[idx]['robot_state'], demo[idx+1]['robot_state'])

        action = np.append(delta_p, gripper) #dx,dy,dz,gripper
        act_traj.append(action)



    exp_traj = {
        "observations": {
            "retrieve_feature": np.array(feature_traj),
            "pixels": np.array(pixel_traj),
        },
        "actions": np.array(act_traj),
    }

    # 1. Print nested observation shapes
    assert len(feature_traj) == len(act_traj)==len(pixel_traj)

    # 2. Print top-level action shape
    print(f"actions: {exp_traj['actions'].shape}")

    return exp_traj

def get_action_matrix(state1,state2):
    pose_delta = state1.O_T_EE.inverse *  state2.O_T_EE
    #combined_action = [None]*7
    delta_p = pose_delta.translation
    #combined_action[3:6] = Rotation.from_quat(pose_delta.quaternion).as_euler("xyz")
    #combined_action[6] = current_state["gripper_command"]
    return delta_p

def save_dataset(folder_path):
    base_dir = Path(folder_path)
    for file_path in base_dir.glob('episode1.npy'):
        print(f"Processing: {file_path.name}")
        processed_data = data_process(file_path)
        new_filename = f"data_{file_path.name}"
        save_path = base_dir / new_filename

        # 4. Save the processed file
        np.save(save_path, processed_data)
        print(f"Saved to: {save_path}")

save_dataset('/home/carol/Project/4-RegIC_IL/ReG_IL_real/expert_demos')