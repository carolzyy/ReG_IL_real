# %%
from franky import *
from scipy.spatial.transform import Rotation
import pyrealsense2 as rs
import cv2
import numpy as np

#--------------- init robot with franky and load OpenVLA-------------------------------
robot = Robot("172.16.0.2")
robot.relative_dynamics_factor = RelativeDynamicsFactor(0.20, 0.40, 0.60)
motion_dynamics_factor = 0.10 #multiplies the above factor during carteisan motion
gripper = Gripper("172.16.0.2")
speed = 0.05  # [m/s]
force = 20.0  # [N]
width = 0.06 #[m]

# ---------------  Move to start --------------------------------------------------------
robot.recover_from_errors()
gripper_open = True
gripper.move_async(width, speed)

init_config = JointMotion([0.001, -0.04124589978198071, 0.001, -2.4789123424790103, 0.001, 2.4785007061817375, 0.785398163397]) #0.0 > 0.001 to avoid errors
robot.move(init_config)


# ----------------- inspect view and algin objects with test scene -------------------------------------------

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640,480, rs.format.bgr8, 30)
pipe.start(cfg)

while(True):
    frame = pipe.wait_for_frames()
    color_frame = frame.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    x_center = int(np.size(color_image,1)/2)
    color_image = color_image[:, x_center-240:x_center+240]

    #TODO: load the first frame of some training episode as cv2_target and uncomment below > manually align images so robot overlaps
    #combined = cv2.addWeighted(color_image,0.5,cv2_target,0.5,0)
    #cv2.imshow('image', combined) 

    cv2.imshow('image', color_image)
        
    #cv2 imshow uses bgr channel ordering, but if your model was trained on rgb you would need to switch channel order here
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    #Example of Cartesian relative motion (end effector delta), assuming your model has generated this:
    #ASSUMPTION THAT MODEL HAS PRODUCED "action" which contains [delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw, grippper(binary)]
    EE_delta = Affine(action[0:3], Rotation.from_euler("xyz", action[3:6]).as_quat()) 
    motion = CartesianMotion(EE_delta, ReferenceType.Relative, relative_dynamics_factor=motion_dynamics_factor)
    robot.move(motion, asynchronous=True) #async = interupts motion with next command when it arrives
    
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()

#%%
robot.recover_from_errors()
robot.move(init_config)
gripper.move(width, speed)
