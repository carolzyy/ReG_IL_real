import numpy as np
from video import VideoRecorder
from pathlib import Path
from franky import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def save_buff_to_map4():
    out_path = Path('/home/carol/Project/4-RegIC_IL/ReG_IL_real/train_video')
    video_recorder = VideoRecorder(out_path)
    # Load the .npz file
    file_path = '/media/carol/KINGSTON/RegIL/logs/03.30_train/regil/214830/expert_buffer.npz'
    data = np.load(file_path)

    # 1. List all available keys in the file
    keys = data.files
    print(f"Keys in the file: {keys}")

    # 2. Analyze the contents of each key
    for key in keys:
        array = data[key]
        print(f"\nAnalysis for key: '{key}'")
        print(f" - Shape: {array.shape}")
        print(f" - Data Type: {array.dtype}")

        # Optional: Basic statistics for numerical data
        if np.issubdtype(array.dtype, np.number):
            print(f" - Min: {np.min(array)}")
            print(f" - Max: {np.max(array)}")
    reward = data['reward']
    done = data['done']
    obs = data['obs_pixels']
    video_recorder.init(obs[0])
    for img in obs[1:]:
        video_recorder.record(img)
    video_recorder.save(f'peg_in.mp4')
    # Close the file if you are done
    data.close()


file_path = '/media/carol/KINGSTON/RegIL/logs/04012250_eval/BC/eval_0_1.npy'
data = np.load(file_path, allow_pickle=True)
print(f'Traj length is {len(data)}, include {data[0].keys()}')
traj_ee = []
demo_ee = []
demo_path = '/media/carol/KINGSTON/RegIL/collect_data/peg-hard/raw_peg-h_13.npy'
demo_data = np.load(demo_path, allow_pickle=True)
for state in data[1:]:
    #observation = state['observation']
    #action = state['action']
    robot_state = state['robot_state']
    traj_ee.append(robot_state.O_T_EE.translation)

for demo_sate in demo_data:
    #demo_obs = demo_sate['observation']
    #demo_action = demo_sate['action']
    demo_robot_state = demo_sate['robot_state']
    demo_ee.append(demo_robot_state.O_T_EE.translation)

# Convert lists to numpy arrays for easier indexing: [N, 3]
traj_ee = np.array(traj_ee)
demo_ee = np.array(demo_ee)

# --- Plotting and Comparison ---
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot Demo: Blue line with circle markers
ax.plot(demo_ee[:, 0], demo_ee[:, 1], demo_ee[:, 2],
        label='Demonstration', color='blue',
        linestyle='-', marker='o', markersize=4, alpha=0.5)

# Plot Executed Trajectory: Red line with 'x' markers
ax.plot(traj_ee[:, 0], traj_ee[:, 1], traj_ee[:, 2],
        label='Executed Trajectory', color='red',
        linestyle='-', marker='x', markersize=4, alpha=0.8)

# Labeling axes
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_title('EE Trajectory Comparison: Points and Lines')
ax.legend()

# Set equal scaling to see the true shape of the movement
# This prevents one axis from looking stretched
ax.set_box_aspect([np.ptp(demo_ee[:,0]), np.ptp(demo_ee[:,1]), np.ptp(demo_ee[:,2])])

plt.show()
