import torch
import torch.nn.functional as F
import numpy as np
from utils.encoders import get_encoders
from pathlib import Path
from scipy.spatial.transform import Rotation
import cv2

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
    motion_traj = []


    for idx in range(len(demo)-1):
        image = demo[idx]['image'] # resize to 128,128 for the buffer size
        image = cv2.resize(image, (84, 84), interpolation=cv2.INTER_AREA)
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
        #motion_traj.append(demo[idx]['motion'])



    exp_traj = {
        "observations": {
            "retrieve_feature": np.array(feature_traj),
            "pixels": np.array(pixel_traj),
        },
        "actions": np.array(act_traj),
        #"motion": np.array(motion_traj),
    }
    save_images_to_mp4(image_list = pixel_traj,
                       output_path=Path('/home/carolzhang/Project/RegIL/ReG_IL_real/dataset/dataset/'),
                       file_name=path.stem+'.mp4')

    # 1. Print nested observation shapes
    assert len(feature_traj) == len(act_traj)==len(pixel_traj)

    # 2. Print top-level action shape
    print(f"actions: {exp_traj['actions'].shape}")

    return exp_traj

def get_action_matrix(state1,state2):
    pose_delta = state1.O_T_EE.inverse *  state2.O_T_EE
    delta_p = pose_delta.translation
    return delta_p

def save_dataset(folder_path):
    base_dir = Path(folder_path)
    for file_path in base_dir.glob('raw_reach*.npy'):
        print(f"Processing: {file_path.name}")
        processed_data = data_process(file_path)
        new_filename = f"dataset_{file_path.name}"
        save_path = Path.cwd() / 'expert_demos' / new_filename

        # 4. Save the processed file
        np.save(save_path, processed_data)
        print(f"Saved to: {save_path}")

from video import VideoRecorder
def save_images_to_mp4(image_list, output_path='./', file_name='output.mp4'):
    """
    Converts a list of images (numpy arrays) into an MP4 video.
    """
    if not image_list:
        print("The image list is empty.")
        return

    # 1. Determine dimensions from the first image
    height, width, layers = image_list[0].shape
    size = (width, height)
    recoder = VideoRecorder(output_path, render_size=width)
    recoder.init(image_list[0])

    for img in image_list:
        # Optional: Standardize size if images vary
        if (img.shape[1], img.shape[0]) != size:
            img = cv2.resize(img, size)

        recoder.record(img)

    recoder.save(file_name)
    print(f"Successfully saved {len(image_list)} frames to {output_path}/output.mp4")

#data = np.load('/expert_demos/data_reach.npy', allow_pickle=True).item()
#print(data.keys())

save_dataset(folder_path='/home/carolzhang/Project/RegIL/ReG_IL_real/dataset/')