import gymnasium as gym
from gymnasium import spaces
import cv2
import numpy as np

import pickle
from scipy.spatial.transform import Rotation
import pyrealsense2 as rs
from franky import *
import time

class Franka():
    def __init__(
        self,
        action_dim = 4,
        gripper_open = True
    ):
        super(Franka, self).__init__()
        self.action_dim = action_dim #（dx,dy,dz,gripper)

        self.n_channels = 3
        self.gripper_open_status = True
        self.motion_dynamics_factor = 0.1
        robot = Robot("172.16.0.2")
        
        self.robot = robot
        gripper = Gripper("172.16.0.2")
        self.gripper_speed = 0.05  # [m/s]
        self.gripper_force = 20.0  # [N]
        self.gripper_width = 0.06  # [m]
        #gripper.move_async(self.gripper_width, self.gripper_speed)
        self.gripper = gripper
        self.init_config = JointMotion(
            [0.001, -0.04124589978198071, 0.001, -2.4789123424790103, 0.001,
             2.4785007061817375, 0.785398163397],#relative_dynamics_factor=0.05
        )
        self.gripper_open_init = gripper_open
        self.pos_range = np.array([0.05, 0.05, 0.03])

    # gripper.asyn_move may lead to problem, so change to move
    def robot_reset(self,random_init=True):
        self.robot.recover_from_errors()
        self.robot.relative_dynamics_factor = 0.05

        if self.gripper_open_init:
            self.gripper.move(width=self.gripper_width,
                                    speed=self.gripper_speed,
                                    )
        else:
            self.gripper.move(width=0.005,
                              speed=self.gripper_speed,
                              )
        time.sleep(0.5)

        self.robot.move(self.init_config)
        if random_init:
            self.randomize_ee_position()

        self.robot.relative_dynamics_factor = RelativeDynamicsFactor(0.20, 0.20, 0.2)
        #print(f'Robot Reset with gripper open {self.gripper_open_init}')


    def robot_act(self,action,asynchronous=True):
        EE_delta = Affine(action[0:3])
        gripper_act = action[-1]
        # A linear motion in Cartesian space relative to the initial position
        motion = CartesianMotion(EE_delta, ReferenceType.Relative,
                                 relative_dynamics_factor=self.motion_dynamics_factor)
        self.robot.move(motion, asynchronous=asynchronous)  # async = interupts motion with next command when it arrives
        if self.gripper_open_init:
            if ((gripper_act < 0.5) and self.gripper_open_status) or (not self.gripper_open_init):
                self.gripper_close()
            elif (gripper_act > 0.5) and (not self.gripper_open_status):
                self.gripper_open()


    def gripper_close(self):
        self.gripper.move(width =0.005,
                                 speed =self.gripper_speed,
                                 #self.gripper_force,
                                 #epsilon_outer=1.0
                                 ) #move_async
        #self.gripper.grasp_async(0.0, self.gripper_speed, self.gripper_force, epsilon_outer=1.0)
        time.sleep(0.1)
        self.gripper_open_status = False

    def gripper_open(self):
        self.gripper.move(
                                 width = self.gripper_width,
                                 speed = self.gripper_speed,
                                 )
        time.sleep(0.1)
        self.gripper_open_status = True
    
    @property
    def robot_mode(self):
        return self.robot.state.robot_mode

    def randomize_ee_position(self):
        """
        Moves the end-effector to a random position within a small
        bounding box relative to the reset position.
        """
        # Generate random offsets: e.g., between -0.05m and +0.05m
        random_offset = np.random.uniform(-self.pos_range, self.pos_range)

        # Create an Affine transformation for the delta
        # We keep rotation (0,0,0) to stay aligned with the reset orientation
        random_delta = Affine(random_offset)

        # Define the motion
        motion = CartesianMotion(
            random_delta,
            ReferenceType.Relative,
            relative_dynamics_factor=0.1
        )

        print(f"Randomizing position by: {random_offset}")
        self.robot.move(motion,asynchronous=False)
        time.sleep(0.2)


class RobotEnv(gym.Env):
    def __init__(
        self,
        gripper_open,
        height=224,
        width=224,
        use_robot=False,  # True when robot used
        max_path_length = 99,
        act_max = [1,1,1],
        act_min = [0,0,0],
        debug_log= '/home/carol/Project/4-RegIC_IL/ReG_IL_real/exp_local/03.17_train/205057/all_retrieve_traj.npz'

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
        self.max_path_length =max_path_length
        self.robot = None
        self.act_stat = {
            'max':np.array(act_max),
            'min':np.array(act_min),
        }
        self.input_action = None
        debug = False


        self.observation_spec = spaces.Box(
            low=0, high=255, shape=(self.n_channels,height, width), dtype=np.uint8
        )
        self.action_spec = spaces.Box(
            low=0.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )

        if self.use_robot:
            self.pipe = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # set resolution and FPS here
            self.pipe.start(cfg)
            self.robot = Franka(
                gripper_open = gripper_open
            )
        elif debug:
            file_path = debug_log
            print(f'Debug from logs:{file_path}')
            data = np.load(file_path)
            self.obs_traj = data['obs_pixels']
            self.act_traj = data['action_policy']
            #self.done_traj = data['done']




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
            #print(f"no robot,excuate action is {action}")
            if getattr(self,"obs_traj",None) is not None:
                obs[f"pixels"] = self.obs_traj[self.episode_step]
                #done = self.done_traj[self.episode_step]
            else:
                obs[f"pixels"] = np.zeros((self.height, self.width, self.n_channels),dtype=np.uint8).transpose(2, 0, 1)
        else:
            self.robot.robot_act(action * 5,)
            obs = self.get_frame()
            #debug for the image
            #save_path = f'/home/carolzhang/Project/RegIL/ReG_IL_real/expert_demos/step_{self.episode_step}.png'
            #cv2.imwrite(save_path, cv2.cvtColor(obs["pixels"], cv2.COLOR_RGB2BGR))
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
    
    def get_observation(self):
        obs = {}
        if self.robot is None:
            print(f"no robot,")
            obs[f"pixels"] = np.zeros((self.height, self.width, self.n_channels),dtype=np.uint8).transpose(2, 0, 1)
        else:
            obs = self.get_frame()

        return obs

    '''
    RealSense get_data()  Always RGB.
    OpenCV Functions (resize, crop)  Don't care (works on either)
    OpenCV Display (imshow) Needs BGR. need be cv2.COLOR_RGB2BGR.
    '''
    def get_frame(self):
        frame = self.pipe.wait_for_frames()
        color_frame = frame.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        x_center = int(np.size(color_image, 1) / 2)
        color_image = color_image[:, x_center - 240:x_center + 240] #480,480
        big_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        color_image = cv2.resize(big_image, (self.width, self.height), interpolation=cv2.INTER_AREA)
        # cv2 imshow uses bgr channel ordering, but if your model was trained on rgb you would need to switch channel order here

        color_image = color_image.transpose(2, 0, 1)

        return {
            'pixels': color_image.copy(),
            'render': big_image.copy()
        }

    def reset(self):  # currently same positions, with gripper opening
        self.episode_step = 0
        if self.use_robot:
            try:
                self.robot.robot_reset()
                print("-----  Robot Reset Process Finished----")
            except Exception as e:
                print(f"!!! Robot Reset Failed: {e} !!!")
                try:
                    print("Attempting automatic error recovery...")
                    self.robot.robot.recover_from_errors()
                    self.robot.robot_reset()
                except:
                    print("Critical Hardware Error: Please check the E-Stop or Network.")
            obs = self.get_frame()
            return obs,False
        else:
            obs = {}
            #obs["features"] = np.zeros(self.feature_dim)
            obs["pixels"] = np.zeros((self.height, self.width, self.n_channels),dtype=np.uint8).transpose(2, 0, 1)
            return obs,False


def make(
    use_robot,
    seed,
    height,
    width,
    max_episode_len,
    eval,
    pixel_keys,
    act_max,
    act_min,
    task
):
    # Convert task_names, which is a list, to a dictionary
    print(f'Init {task} env')
    if 'reach' in task[0]:
        gripper_open = False
        print(f'close gripper')
    else:
        gripper_open = True
    env = RobotEnv(
                   height=height,
                   width=width,
                   use_robot=use_robot,
                   max_path_length=max_episode_len,
                   act_max=act_max,
                   act_min=act_min,
                   gripper_open = gripper_open
                   )

    return env