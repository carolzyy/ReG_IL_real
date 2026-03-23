import numpy as np

# Load the .npz file
file_path = '/home/carol/Project/4-RegIC_IL/ReG_IL_real/exp_local/03.17_train/205057/all_retrieve_traj.npz'
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
        print(f" - Mean: {np.mean(array)}")
reward = data['reward']
done = data['done']
obs = data['obs_pixels']
# Close the file if you are done
data.close()
