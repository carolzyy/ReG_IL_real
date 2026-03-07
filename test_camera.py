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