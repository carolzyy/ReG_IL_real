import gym
from gym import spaces
import cv2
import numpy as np

import pickle
from scipy.spatial.transform import Rotation
import pyrealsense2 as rs
from franky import *
from utils import manual_open_mouse

def get_quaternion_orientation(cartesian):
    """
    Get quaternion orientation from axis angle representation
    """
    pos = cartesian[:3]
    ori = cartesian[3:]
    r = Rotation.from_rotvec(ori)
    quat = r.as_quat()
    return np.concatenate([pos, quat], axis=-1)

class Robot():
    def __init__(
        self,
        height=224,
        width=224,
        use_camera=True,
        action_dim = 4,
        use_mouse=False,
    ):
        super(Robot, self).__init__()
        self.height = height
        self.width = width
        self.use_camera = use_camera
        self.action_dim = action_dim #（dx,dy,dz,gripper)

        self.n_channels = 3
        self.recording_frequency = 30
        self.gripper_open = True
        if use_mouse:
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
        gripper.move_async(self.gripper_width, self.gripper_speed)
        self.gripper = gripper
        self.init_config = JointMotion(
            [0.001, -0.04124589978198071, 0.001, -2.4789123424790103, 0.001,
             2.4785007061817375, 0.785398163397]
        )

    def robot_reset(self):
        self.robot.recover_from_errors()
        self.robot.move(self.init_config)
        self.gripper_open()


    def robot_act(self,action):
        EE_delta = Affine(action[0:3])
        gripper_act = action[-1]
        # A linear motion in Cartesian space relative to the initial position
        motion = CartesianMotion(EE_delta, ReferenceType.Relative,
                                 relative_dynamics_factor=self.motion_dynamics_factor)
        self.robot.move(motion, asynchronous=True)  # async = interupts motion with next command when it arrives
        if gripper_act < 0:
            self.gripper_close()
        else:
            self.gripper_open()


    def gripper_close(self):
        self.gripper.grasp_async(0.0,
                                 self.gripper_speed,
                                 self.gripper_force, epsilon_outer=1.0)

    def gripper_open(self):
        self.gripper.grasp_async(
                                 self.gripper_width,
                                 self.gripper_speed,
                                 )


class RobotEnv(gym.Env):
    def __init__(
        self,
        height=224,
        width=224,
        use_robot=True,  # True when robot used
    ):
        super(RobotEnv, self).__init__()
        self.height = height
        self.width = width
        self.use_robot = use_robot
        self.feature_dim = 8
        self.action_dim = 5 #（dx,dy,dz,gripper)

        self.n_channels = 3
        self.reward = 0
        self.recording_frequency = 30
        self.gripper_open = True
        self.motion_dynamics_factor = 0.10

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(height, width, self.n_channels), dtype=np.uint8
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )

        if self.use_robot:
            self.pipe = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # set resolution and FPS here
            self.pipe.start(cfg)
            self.robot = Robot()

    def step(self, action):
        print("current step's action is: ", action)

        action = np.array(action)

        self.robot.act(action)

        # subscribe images
        image_list = []
        for subscriber in self.image_subscribers:
            image_list.append(subscriber.recv_rgb_image()[0])

        obs = {}
        obs["features"] = np.array(robot_state, dtype=np.float32)
        for idx, image in enumerate(image_list):
            obs[f"pixels{idx}"] = cv2.resize(image, (self.width, self.height))

        return obs, self.reward, False, None

    def reset(self):  # currently same positions, with gripper opening
        if self.use_robot:
            print("resetting")
            self.action_request_socket.send(b"reset")
            reset_state = pickle.loads(self.action_request_socket.recv())

            # subscribe robot state
            self.action_request_socket.send(b"get_state")
            robot_state = pickle.loads(self.action_request_socket.recv())["xarm"]
            cartesian = robot_state[:6]
            quat_cartesian = get_quaternion_orientation(cartesian)
            robot_state = np.concatenate([quat_cartesian, robot_state[6:]], axis=0)

            # subscribe images
            image_list = []
            for subscriber in self.image_subscribers:
                image_list.append(subscriber.recv_rgb_image()[0])

            obs = {}
            obs["features"] = robot_state
            for idx, image in enumerate(image_list):
                obs[f"pixels{idx}"] = cv2.resize(image, (self.width, self.height))

            return obs
        else:
            obs = {}
            obs["features"] = np.zeros(self.feature_dim)
            obs["pixels"] = np.zeros((self.height, self.width, self.n_channels))
            return obs

    def render(self, mode="rgb_array", width=640, height=480):
        print("rendering")
        # subscribe images
        image_list = []
        for subscriber in self.image_subscribers:
            image = subscriber.recv_rgb_image()[0]
            image_list.append(cv2.resize(image, (width, height)))

        obs = np.concatenate(image_list, axis=1)
        return obs


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