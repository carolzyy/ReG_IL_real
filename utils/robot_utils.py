from pyspacemouse import get_device_specs
from pyspacemouse.device import SpaceMouseDevice
import hid

def manual_open_mouse():
    spec = get_device_specs()
    mouse = SpaceMouseDevice(info=spec['SpaceMouseCompact'])

    vendor_id = 0x256f  # 3Dconnexion
    product_id = 0xc635 # SpaceMouse Compact

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

    return mouse

import numpy as np
from scipy.spatial.transform import Rotation as R


def wxyz_to_xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]])

def get_delta_q(q1,q2,wxyz=True):
    if wxyz:
        q1 = wxyz_to_xyzw(q1)
        q2 = wxyz_to_xyzw(q2)

    # 2. Load into SciPy
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)

    # 3. Calculate relative rotation: r_rel = r2 * r1_inv
    # In SciPy, the * operator performs composition: r_total = r_second * r_first
    r_rel = r2 * r1.inv()

    # 4. Output the result back in [w, x, y, z]
    rel_quat_xyzw = r_rel.as_quat()

    if wxyz:
        rel_quat_wxyz = np.array([rel_quat_xyzw[3], rel_quat_xyzw[0], rel_quat_xyzw[1], rel_quat_xyzw[2]])

    print(f"Relative Rotation (WXYZ): {rel_quat_wxyz}")

    return rel_quat_wxyz


def get_action(state1,state2):
    ee_relative_p1 = state1.O_T_EE.translation
    ee_relative_quat1 = state1.O_T_EE.quaternion
    ee_relative_p2 = state2.O_T_EE.translation
    ee_relative_quat2 = state2.O_T_EE.quaternion
    delta_p = ee_relative_p1 - ee_relative_p2
    delta_quat = get_delta_q(ee_relative_quat1,ee_relative_quat2)
    delta_act=np.array([delta_p, delta_quat])
    return delta_act