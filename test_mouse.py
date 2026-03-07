import hid
import pyspacemouse
import time

def force_open_spacemouse():
    vendor_id = 0x256f  # 3Dconnexion
    product_id = 0xc635 # SpaceMouse Compact
    
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
         
        return h 
        
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

def force_combined_open():
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
        return None

    print(f"Found SpaceMouse at: {target_path}")

    # 2. Open the HID device manually first
    h = hid.device()
    h.open_path(target_path)
    
    # 3. Wrap that raw HID object in the pyspacemouse interface
    # This bypasses the library's "auto-search" which was failing you
    dev = pyspacemouse.Device(h)
    print("Successfully linked raw device to pyspacemouse!")
    return dev

from pyspacemouse import get_device_specs
from pyspacemouse.device import SpaceMouseDevice
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
gripper_open = False

while mouse:
    recording = True
    while recording:
        state = mouse.read()
        print(state.buttons)

        if state.buttons[0]:
            gripper_open = False
            print('gripper_open = False')
        elif state.buttons[1]:
            gripper_open = True
            print('gripper_open = True')
        elif state.buttons[0] and state.buttons[1]:
            recording = False
            print('recording = False')
            break

        time.sleep(0.1)