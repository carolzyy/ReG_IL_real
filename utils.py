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
