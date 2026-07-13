import cv2
import numpy as np
from numpy.typing import NDArray

from config import IMAGE_SIZE


def center_crop_square(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Crop the largest centered square; do not stretch the image."""
    height, width = image.shape[:2]
    side = min(height, width)
    y0 = (height - side) // 2
    x0 = (width - side) // 2
    return image[y0 : y0 + side, x0 : x0 + side]


def frame_to_sample(frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
    if frame.ndim == 2:
        gray = frame
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cropped = center_crop_square(gray)
    return cv2.resize(cropped, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)


def sample_to_features(sample: NDArray[np.uint8]) -> NDArray[np.float32]:
    return sample.flatten().astype(np.float32) / 255.0
