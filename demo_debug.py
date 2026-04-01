import numpy as np
from video import VideoRecorder
from pathlib import Path
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
