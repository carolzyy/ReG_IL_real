import gymnasium as gym
from gymnasium import spaces
import cv2
import numpy as np

import pickle
from scipy.spatial.transform import Rotation
import pyrealsense2 as rs
from franky import *


def get_quaternion_orientation(cartesian):
    """
    Get quaternion orientation from axis angle representation
    """
    pos = cartesian[:3]
    ori = cartesian[3:]
    r = Rotation.from_rotvec(ori)
    quat = r.as_quat()
    return np.concatenate([pos, quat], axis=-1)

class Franka():
    def __init__(
        self,
        action_dim = 4,
        use_mouse=False,
    ):
        super(Franka, self).__init__()
        self.action_dim = action_dim #（dx,dy,dz,gripper)

        self.n_channels = 3
        self.recording_frequency = 30
        self.gripper_open_status = True
        self.motion_dynamics_factor = 0.05
        if use_mouse:
            from utils.robot_utils import manual_open_mouse
            self.mouse = manual_open_mouse()
            recording_frequency = 30
            linear_scaling = 0.15
            # rot_scaling = 0.15
            roll_scaling = 0.10
            pitch_scaling = 0.10
            yaw_scaling = 0.20
        robot = Robot("172.16.0.2")
        robot.relative_dynamics_factor = RelativeDynamicsFactor(0.20, 0.40, 0.60)
        self.robot = robot
        gripper = Gripper("172.16.0.2")
        self.gripper_speed = 0.05  # [m/s]
        self.gripper_force = 20.0  # [N]
        self.gripper_width = 0.06  # [m]
        #gripper.move_async(self.gripper_width, self.gripper_speed)
        self.gripper = gripper
        self.init_config = JointMotion(
            [0.001, -0.04124589978198071, 0.001, -2.4789123424790103, 0.001,
             2.4785007061817375, 0.785398163397]
        )

    def robot_reset(self):
        print('Robot start reset.......')
        self.robot.recover_from_errors()
        self.robot.move(self.init_config)
        self.gripper_open()
        print('Robot ready!!!!')


    def robot_act(self,action):
        EE_delta = Affine(action[0:3])
        gripper_act = action[-1]
        # A linear motion in Cartesian space relative to the initial position
        motion = CartesianMotion(EE_delta, ReferenceType.Relative,
                                 relative_dynamics_factor=self.motion_dynamics_factor)
        self.robot.move(motion, asynchronous=True)  # async = interupts motion with next command when it arrives
        if (gripper_act < 0.5) and self.gripper_open_status:
            self.gripper_close()
        elif (gripper_act > 0.5) and (not self.gripper_open_status):
            self.gripper_open()


    def gripper_close(self):
        self.gripper.move_async(0.0,
                                 self.gripper_speed,
                                 self.gripper_force,
                                 epsilon_outer=1.0)
        self.gripper_open_status = False

    def gripper_open(self):
        self.gripper.move_async(
                                 width = self.gripper_width,
                                 speed = self.gripper_speed,
                                 )
        self.gripper_open_status = True
    
    @property
    def robot_mode(self):
        return self.robot.state.robot_mode


class RobotEnv(gym.Env):
    def __init__(
        self,
        height=224,
        width=224,
        use_robot=True,  # True when robot used
        max_path_length = 99,
        act_max = [1,1,1],
        act_min = [0,0,0]
    ):
        super(RobotEnv, self).__init__()
        self.height = height
        self.width = width
        self.use_robot = use_robot
        self.feature_dim = 8
        self.action_dim = 4 #（dx,dy,dz,gripper)
        self.episode_step = 0

        self.n_channels = 3
        self.reward = 0
        self.recording_frequency = 30
        self.gripper_open = True
        self.motion_dynamics_factor = 0.10
        self.max_path_length =max_path_length
        self.robot = None
        self.act_stat = {
            'max':np.array(act_max),
            'min':np.array(act_min),
        }
        self.input_action = None


        self.observation_spec = spaces.Box(
            low=0, high=255, shape=(height, width, self.n_channels), dtype=np.uint8
        )
        self.action_spec = spaces.Box(
            low=0.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )

        if self.use_robot:
            self.pipe = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # set resolution and FPS here
            self.pipe.start(cfg)
            self.robot = Franka()

    def act_preprocess(self,action):
        act_min = self.act_stat['min']
        act_range = self.act_stat['max'][:3] - self.act_stat['min'][:3]
        action_xyz = 2*(action[..., :3]-act_min[:3])/(act_range+ 1e-8)-1
        action_gripper =  np.where(action[..., 3:] > 0.5, 1.0, -1)
        action_processed = np.concatenate([action_xyz, action_gripper], axis=-1)
        return action_processed

    def act_posprocess(self,action):
        #act_max = self.act_stat['max']
        act_min = self.act_stat['min']
        act_range = self.act_stat['max'][:3] - self.act_stat['min'][:3]
        action_xyz =  (action[..., :3] + 1.0 )* act_range/2.0 + act_min[:3]
        action_gripper = np.where(action[..., 3:] > 0, 1.0, 0.0)
        return np.concatenate([action_xyz, action_gripper], axis=-1)


    def step(self, action):
        #print(f"current {self.episode_step}th action is: ", action)
        obs = {}
        if self.episode_step == self.max_path_length:
            print(f"current step {self.episode_step} meet the max{self.max_path_length} ")
            done = True
        else:
            done = False

        self.input_action = action.copy()

        action = self.act_posprocess(action)
        if self.robot is None:
            print(f"no robot,excuate action is {action}")
            obs[f"pixels"] = np.zeros((self.height, self.width, self.n_channels),dtype=np.uint8)
        else:
            self.robot.robot_act(action)
            obs["pixels"] = self.get_frame()
            save_path = f'/home/carolzhang/Project/RegIL/ReG_IL_real/expert_demos/step_{self.episode_step}.png'
            cv2.imwrite(save_path, cv2.cvtColor(obs["pixels"], cv2.COLOR_RGB2BGR))
        #print(f'Excuated action is {action}')

        self.episode_step += 1

        return obs, done #, None #obs, reward, done, info

    def get_done(self):
        mode = self.robot.robot_mode
        done = "Stopped" in str(mode) or "Reflex" in str(mode)
        if "Reflex" in str(mode):
            terminted = input('the robot collision,press the button:')
            self.robot.robot.recover_from_errors()

        return done,mode

    def get_frame(self):
        frame = self.pipe.wait_for_frames()
        color_frame = frame.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        x_center = int(np.size(color_image, 1) / 2)
        color_image = color_image[:, x_center - 240:x_center + 240] #480,480
        color_image = cv2.resize(color_image, (self.width, self.height), interpolation=cv2.INTER_AREA)
        # cv2 imshow uses bgr channel ordering, but if your model was trained on rgb you would need to switch channel order here
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        return color_image

    def reset(self):  # currently same positions, with gripper opening
        self.episode_step = 0
        if self.use_robot:
            print("resetting")
            self.robot.robot_reset()

            obs = {}
            obs["pixels"] = self.get_frame()

            return obs,False
        else:
            obs = {}
            #obs["features"] = np.zeros(self.feature_dim)
            obs["pixels"] = np.zeros((self.height, self.width, self.n_channels),dtype=np.uint8)
            return obs,False

    def render(self, mode="rgb_array", width=640, height=480):
        print("rendering")
        obs["pixels"] = self.get_frame()
        return obs

def make(
    frame_stack,
    action_repeat,
    seed,
    height,
    width,
    max_episode_len,
    eval,
    pixel_keys,
    act_max,
    act_min
):
    # Convert task_names, which is a list, to a dictionary
    #tasks = tasks
    env = RobotEnv(
                   height=height,
                   width=width,
                   use_robot=True,
                   max_path_length=max_episode_len,
                   act_max=act_max,
                   act_min=act_min,
                   )

    return env


if __name__ == "__main__":
    env = RobotEnv()
    obs = env.reset()

    for i in range(30):
        action = obs["features"]
        action[0] += 2
        obs, reward, done, _ = env.step(action)

    for i in range(30):
        action = obs["features"]
        action[1] += 2
        obs, reward, done, _ = env.step(action)

    for i in range(30):
        action = obs["features"]
        action[2] += 2
        obs, reward, done, _ = env.step(action)


# check gripper