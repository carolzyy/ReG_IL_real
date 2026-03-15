import numpy as np
import os
import cv2
from retriever import get_retriever



import numpy as np
file_path = '/home/carol/Project/4-RegIC_IL/ReG_IL_real/data/episode1.npy'
demo = np.load(file_path,allow_pickle=True)
print(f"Successfully loaded: {file_path} with length{len(demo)}")
