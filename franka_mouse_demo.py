from franky import *
import pyspacemouse
import time
import pyrealsense2 as rs
import numpy as np
from scipy.spatial.transform import Rotation
import cv2
import threading
import queue
import glob
import re
from pathlib import Path
import argparse
from utils.robot_utils import manual_open_mouse

q = queue.Queue(maxsize=1) #only save the latest image
def show_camera():
    while True:
        img = q.get()
        cv2.imwrite("camera.jpg", img)
threading.Thread(target=show_camera, daemon=True).start()

dataset_path = '/home/carolzhang/Project/RegIL/ReG_IL_real/dataset' 
if not Path(f"{dataset_path}").is_dir():
    Path(f"{dataset_path}").mkdir(parents=True, exist_ok=True)

robot = Robot("172.16.0.2")
robot.relative_dynamics_factor = RelativeDynamicsFactor(0.20, 0.40, 0.60)
motion_dynamics_factor = 0.05 #multiplies the above factor during carteisan motion
gripper = Gripper("172.16.0.2")
speed = 0.05  # [m/s]
force = 20.0  # [N]
width = 0.06 #[m]

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640,480, rs.format.bgr8, 30) #set resolution and FPS here
pipe.start(cfg)

recording_frequency = 30
linear_scaling = 0.15
#rot_scaling = 0.15
roll_scaling = 0.10
pitch_scaling = 0.10
yaw_scaling = 0.20

gripper_open = True
gripper.move_async(width, speed)

episode = []

robot.recover_from_errors()

init_config = JointMotion([0.001, -0.04124589978198071, 0.001, -2.4789123424790103, 0.001, 2.4785007061817375, 0.785398163397]) #0.0 > 0.001 to avoid errors
robot.move(init_config)

#get an image of the initial scene so that the operator can verify that the relevant objects are visible
frame = pipe.wait_for_frames()
color_frame = frame.get_color_frame()
color_image = np.asanyarray(color_frame.get_data())
x_center = int(np.size(color_image,1)/2)
color_image = color_image[:, x_center-240:x_center+240]
q.put(color_image)

#language_instruction = input("Input the language instruction: ")
parser = argparse.ArgumentParser()  
parser.add_argument("task_name", type=str)
task_name = parser.parse_args().task_name

input("Start the demo")
mouse = manual_open_mouse()

if mouse:
    start_time = time.time()
    last_time = start_time
    recording = True
    while recording:
        state = mouse.read()

        noise = np.random.normal(0,0.1,6)

        if not state.buttons[0] and state.buttons[1] and gripper_open:
            gripper.grasp_async(0.0, speed, force, epsilon_outer=1.0)
            gripper_open = False
        elif state.buttons[0] and not state.buttons[1] and not gripper_open:
            #gripper.open_async(speed)
            gripper.move_async(width, speed)
            gripper_open = True
        elif state.buttons[0] and state.buttons[1]:
            recording = False
            motion = CartesianVelocityMotion(Twist([0.0, 0.0, 0.0]), relative_dynamics_factor=motion_dynamics_factor)
            robot.move(motion, asynchronous=True)
            break

        #NOTE: linear delta in base frame, rotation in ee frame
        base_delta_linear = [linear_scaling*(state.y+noise[1]), -linear_scaling*(state.x+noise[0]), linear_scaling*(state.z+noise[2])] #axes are flipped so that they align with starting pose when spacemouse has cable going to the right = same direction as gripperc
        #EE_delta_linear = Rotation.from_quat(robot.state.O_T_EE.quaternion).apply(base_delta_linear)
        EE_delta_rot = Rotation.from_euler("xyz", [roll_scaling*(state.roll+noise[3]), pitch_scaling*(state.pitch+noise[4]), -yaw_scaling*(state.yaw+noise[5])]).as_euler("xyz") #RPY is applied as EE frame rotations directly, axis signs flipped so that spacemouse is aligned with gripper down in typical starting pose

        #end effector Cartesian velocity motion with linear (first argument) and angular (second argument)
        #motion = CartesianVelocityMotion(Twist(base_delta_linear, EE_delta_rot), relative_dynamics_factor=motion_dynamics_factor)
        motion = CartesianVelocityMotion(Twist(base_delta_linear),
                                         relative_dynamics_factor=motion_dynamics_factor)

        robot.move(motion, asynchronous=True)
        
        if time.time()-last_time >= 1/recording_frequency:
            #start_processing = time.time()
            frame = pipe.wait_for_frames()
            color_frame = frame.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            x_center = int(np.size(color_image,1)/2)
            color_image = color_image[:, x_center-240:x_center+240]
            resize_img = cv2.resize(color_image, (128, 128), interpolation=cv2.INTER_AREA)
            q.put(resize_img)
            step = {}
            step["robot_state"] = robot.state #save full state instead of just O_T_EE, because size is neglible compared to image data
            #step["gripper_state"] = gripper.state #NOTE: this actually takes takes quite some time as getting the GRIPPER STATE which is not realtime!!
            step["gripper_command"] = gripper_open #use commanded gripper state instead
            step["motion"] = np.array(base_delta_linear)
            step["image"] = color_image.copy()

            episode.append(step)
            last_time = time.time()

        time.sleep(0.001)
    
    pipe.stop()

    save_ep = input("Save episode (Y/N)?")

    if save_ep.upper() == "Y":
        ids = [int(re.search("episode(.*).npy", file).group(1)) for file in glob.glob(f"{dataset_path}/episode*.npy")]
        ep_id = max(ids)+1 if ids else 0
        np.save(f"{dataset_path}/raw_{task_name}_{ep_id}.npy", episode)
        print(f"Demonstration saved in {dataset_path}/{task_name}_{ep_id}.npy")
    else:
        print("Demonstration not saved")

    gripper.move_async(width, speed)
    robot.move(init_config)

