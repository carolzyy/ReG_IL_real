import numpy as np
import os
import cv2




def save_images_to_mp4(image_list, output_filename='output.mp4', fps=30):
    """
    Converts a list of images (numpy arrays) into an MP4 video.
    """
    if not image_list:
        print("The image list is empty.")
        return

    # 1. Determine dimensions from the first image
    height, width, layers = image_list[0].shape
    size = (width, height)

    # 2. Define the codec and create VideoWriter object
    # 'mp4v' is widely compatible with .mp4 containers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, size)

    for img in image_list:
        # Optional: Standardize size if images vary
        if (img.shape[1], img.shape[0]) != size:
            img = cv2.resize(img, size)

        # Note: OpenCV uses BGR. If your images are RGB (PIL/Matplotlib),
        # uncomment the next line:
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        out.write(img)

    out.release()
    print(f"Successfully saved {len(image_list)} frames to {output_filename}")
