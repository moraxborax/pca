from preprocess import frame_to_sample
from stream import Camera

if __name__ == "__main__":
    camera = Camera(0)
    ret, frame = camera.read()
    if ret and frame is not None:
        sample = frame_to_sample(frame)
        print(f"Captured frame {frame.shape} -> sample {sample.shape}")
    else:
        print("Failed to capture frame")
    camera.release()
