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
    from franky import *

    robot = Robot("172.16.0.2")  # Replace this with your robot's IP

    # Let's start slow (this lets the robot use a maximum of 5% of its velocity, acceleration, and jerk limits)
    robot.relative_dynamics_factor = 0.05

    init_config = JointMotion([0.001, -0.04124589978198071, 0.001, -2.4789123424790103, 0.001, 2.4785007061817375,
                               0.785398163397])  # 0.0 > 0.001 to avoid errors
    robot.move(init_config)

    # Move the robot 20cm along the relative X-axis of its end-effector
    # motion = CartesianMotion(Affine([0.2, 0.0, 0.0]), ReferenceType.Relative)
    # robot.move(motion)
    print('Back to the initial pose')

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