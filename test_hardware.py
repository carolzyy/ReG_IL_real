from franky import *
import time
def test_camera():
    print("-----Testing the hardware of RealSense-----------")
    import pyrealsense2 as rs
    import numpy as np
    import cv2

    pipeline = rs.pipeline()
    config = rs.config()

    # Try a very safe resolution first
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

    print("Attempting to start RealSense...")
    active_pipeline = False

    try:
        pipeline.start(config)
        active_pipeline = True
        print("Stream started successfully!")

        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            img = np.asanyarray(color_frame.get_data())
            cv2.imshow('RealSense RGB', img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error: {e}")
        print("\nPRO-TIP: Check if the camera is in a Blue USB 3.0 port.")

    finally:
        if active_pipeline:
            pipeline.stop()
        cv2.destroyAllWindows()



def test_franka():
    print("-----Testing robot connection-----------")
    #from franky import *


    robot = Robot("172.16.0.2")  # Replace this with your robot's IP

    # Let's start slow (this lets the robot use a maximum of 5% of its velocity, acceleration, and jerk limits)
    robot.relative_dynamics_factor = 0.05

    init_config = JointMotion([0.001, -0.04124589978198071, 0.001, -2.4789123424790103, 0.001, 2.4785007061817375,
                               0.785398163397])  # 0.0 > 0.001 to avoid errors
    robot.move(init_config)
    # In franky/libfranka, RobotMode.UserStopped is typically 4
    #state = robot.state
    #print(state.robot_mode)
    gripper = Gripper("172.16.0.2")
    speed = 0.05  # [m/s]
    force = 20.0  # [N]
    width = 0.06 #[m]
    #gripper.move(width, speed)
    ready = input('Get the cable ready:')
    time.sleep(0.5)
    gripper.grasp_async(0.00, speed/2, force, epsilon_outer=1.0)

    #motion1 = CartesianMotion(Affine([0.2, 0.0, 0.0]), ReferenceType.Relative)
    #robot.move(motion1, asynchronous=False)

    #time.sleep(0.5)
    # Note that, similar to reactions, when preempting active motions with new motions, the
    # control mode cannot change. Hence, we cannot use, e.g., a JointMotion here.
    #motion2 = CartesianMotion(Affine([-0.2, 0.0, 0.0]), ReferenceType.Relative)
    #robot.move(motion2, asynchronous=False)

    # Move the robot 20cm along the relative X-axis of its end-effector
    motion = CartesianMotion(Affine([-0.0, 0.0, -0.05]), ReferenceType.Relative)
    robot.move(motion)
    state = robot.state
    joint_state = robot.current_joint_state
    joint_pos = joint_state.position
    print('Back to the initial pose',joint_state)


def check_franka_interface(robot_ip="172.16.0.2"):
    print("-----Testing robot connection-----------")
    import time
    try:
        # Initialize the robot
        print(f"Connecting to robot at {robot_ip}...")
        robot = Robot(robot_ip)
        print("Connected! Press the buttons on the robot to test them.")
        print("Press Ctrl+C to exit this script.")
        print("-" * 40)

        last_mode = False
        robot.relative_dynamics_factor = 0.05

        motion1 = CartesianMotion(Affine([0.1, 0.0, 0.0]), ReferenceType.Relative)
        robot.move(motion1, asynchronous=False)

        time.sleep(0.5)
        # Note that, similar to reactions, when preempting active motions with new motions, the
        # control mode cannot change. Hence, we cannot use, e.g., a JointMotion here.
        motion2 = CartesianMotion(Affine([-0.1, 0.0, 0.0]), ReferenceType.Relative)
        robot.move(motion2, asynchronous=False)

        while True:
            # 1. Check the White User Button
            current_mode = robot.state.robot_mode

            if current_mode != last_mode:
                print(f"[MODE] Status changed from {last_mode} to: {current_mode}")
                last_mode = current_mode
            
            robot.move(motion1,asynchronous=False)
            time.sleep(0.5)
            robot.move(motion2,asynchronous=False)

            # If E-Stop is pressed, mode usually switches to UserStopped or Reflex
            if "Stopped" in str(current_mode):
                print(f"[SYSTEM ALERT] Robot is in {current_mode} mode!")
            elif "Stopped" in str(last_mode) or "Reflex" in str(last_mode):
                print(f"[SYSTEM ALERT] Robot is recoverying")
                robot.recover_from_errors()
                init_config = JointMotion([0.001, -0.04124589978198071, 0.001, -2.4789123424790103, 0.001, 2.4785007061817375, 0.785398163397]) #0.0 > 0.001 to avoid errors
                robot.move(init_config)

            # Small sleep to prevent CPU saturation
            time.sleep(0.5)
    except ControlException as e:
            print(f'{e}')
            current_mode = robot.state.robot_mode

def check_btn_log(robot_ip="172.16.0.2"):
    print("-----Testing robot connection-----------")
    robot = Robot(robot_ip)
    robot.recover_from_errors()
    robot.relative_dynamics_factor = 0.05

    init_config = JointMotion([0.001, -0.04124589978198071, 0.001, -2.4789123424790103, 0.001, 2.4785007061817375,
                               0.785398163397])  # 0.0 > 0.001 to avoid errors
    robot.move(init_config)
    motion1 = CartesianMotion(Affine([0.2, 0.0, 0.0]), ReferenceType.Relative)
    motion2 = CartesianMotion(Affine([-0.2, 0.0, 0.0]), ReferenceType.Relative)
    while True:  # 最外层循环，保证程序报错后不退出
        try:
            # 1. 正常的运动逻辑
            print("正在执行运动任务...")
            robot.move(motion1, asynchronous=False)
            time.sleep(0.5)
            robot.move(motion2, asynchronous=False)
        except ControlException as e:
            print(f"Error detected: {e}")
            while "Stopped" in str(robot.state.robot_mode) or "Reflex" in str(robot.state.robot_mode):
                time.sleep(0.2)  #maybe have some problem with the status changing

            print("Button released, try recovery")

            try:
                success_input = input("Success or not(Y/N):")
                success = (success_input.upper() == "Y")
                print(f'This episode ended with {success},robot start reset')
                robot.recover_from_errors()
                time.sleep(0.5)

                init_config = JointMotion([0.001, -0.041, 0.001, -2.478, 0.001, 2.478, 0.785])
                robot.move(init_config)
                print("robot recoveried")
            except Exception as recovery_e:
                print(f"恢复失败: {recovery_e}，将重新尝试...")

def test_mouse_hid():
    print("-----Testing mouse data read with hid library-----------")
    import hid
    import pyspacemouse
    import time

    vendor_id = 0x256f  # 3Dconnexion
    product_id = 0xc635  # SpaceMouse Compact

    print("Searching for SpaceMouse Compact...")
    target_path = None

    # Find the specific path for the Compact model
    for device in hid.enumerate():
        if device['vendor_id'] == vendor_id and device['product_id'] == product_id:
            target_path = device['path']
            print(f"Found it at path: {target_path}")
            break

    if not target_path:
        print("Device not found in enumeration.")
        return

    try:
        # Create a raw HID device object
        h = hid.device()
        h.open_path(target_path)
        print("Success! Manually opened the device path.")

        print("Reading data (Press Ctrl+C to stop)...")
        while True:
            # SpaceMouse Compact usually sends 7 or 13 byte packets
            d = h.read(64)
            if d:
                print(f"Raw Data: {d}")
            time.sleep(0.01)

    except Exception as e:
        print(f"Failed to open path {target_path}: {e}")
        print("\nPossible Reason: The device is 'busy' (claimed by another driver).")
    finally:
        h.close()


def test_spacelib_hid():
    print("-----Testing spacelib connection-----------")
    import hid
    import time
    from pyspacemouse import get_device_specs
    from pyspacemouse.device import SpaceMouseDevice
    spec = get_device_specs()
    mouse = SpaceMouseDevice(info=spec['SpaceMouseCompact'])

    vendor_id = 0x256f  # 3Dconnexion
    product_id = 0xc635  # SpaceMouse Compact

    print("Enumerating HID devices...")
    target_path = None

    # 1. Manually find the path
    for device_dict in hid.enumerate():
        if device_dict['vendor_id'] == vendor_id and device_dict['product_id'] == product_id:
            target_path = device_dict['path']
            break

    if not target_path:
        print("Device not found in HID enumeration.")
    h = hid.device()
    h.open_path(target_path)
    mouse._device = h

    while mouse:
        recording = True
        while recording:
            state = mouse.read()
            print(state.buttons)
            if state.buttons[0]:
                print('Press Button 0')
            elif state.buttons[1]:
                print('Press Button 1')
            elif state.buttons[0] and state.buttons[1]:
                print('Press Button 0 and 1 at the same time')
                break

            time.sleep(0.1)


import time
import numpy as np
def test_demo():
    robot_ip="172.16.0.2"
    data = np.load('/home/carolzhang/Project/RegIL/ReG_IL_real/expert_demos/dataset_matrix_reach_0.npy',allow_pickle=True).item()
    action = data['actions']
    motion = data['motion']

    print("-----Testing robot connection-----------")
    # Initialize the robot
    print(f"Connecting to robot at {robot_ip}...")
    robot = Robot(robot_ip)
    print("Connected! Press the buttons on the robot to test them.")
    print("Press Ctrl+C to exit this script.")
    print("-" * 40)
    robot.relative_dynamics_factor = 0.05
    robot.recover_from_errors()
    init_config = JointMotion(
            [0.001, -0.04124589978198071, 0.001, -2.4789123424790103, 0.001,
             2.4785007061817375, 0.785398163397]
        )
    robot.move(init_config, asynchronous=False)
    time.sleep(1)

    last_mode = False
    robot.relative_dynamics_factor = 0.05
    print(f"Start testing")

    for act in action:
        #motion1 = CartesianMotion(Affine(act[:3]), ReferenceType.Relative)
        # A linear motion in Cartesian space relative to the initial position
        motion = CartesianMotion(Affine(act[:3]), ReferenceType.Relative,
                                 #relative_dynamics_factor=0.05
                                 )
        #robot.move(act, asynchronous=False)
        robot.move(motion, asynchronous=True)
        time.sleep(1/20)

test_franka()
#test_camera()