from franky import *

robot = Robot("172.16.0.2")  # Replace this with your robot's IP

# Let's start slow (this lets the robot use a maximum of 5% of its velocity, acceleration, and jerk limits)
robot.relative_dynamics_factor = 0.05

init_config = JointMotion([0.001, -0.04124589978198071, 0.001, -2.4789123424790103, 0.001, 2.4785007061817375, 0.785398163397]) #0.0 > 0.001 to avoid errors
robot.move(init_config)

# Move the robot 20cm along the relative X-axis of its end-effector
#motion = CartesianMotion(Affine([0.2, 0.0, 0.0]), ReferenceType.Relative)
#robot.move(motion)

# Move the robot -20cm along the relative X-axis of its end-effector
#motion2 = CartesianMotion(Affine([-0.2, 0.0, 0.0]), ReferenceType.Relative)
#robot.move(motion2)
print('Back to the initial pose')
