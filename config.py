from pathlib import Path

IMAGE_SIZE = 64
N_COMPONENTS = 128
SAMPLE_RATE_HZ = 10
N_FEATURES = IMAGE_SIZE * IMAGE_SIZE

CLASSES = ["forward", "stop", "left", "right", "search"]
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASSES)}

DATA_DIR = Path("data")
ARTIFACTS_DIR = Path("artifacts")

KEY_TO_STATE = {
    "w": "forward",
    "s": "stop",
    "a": "left",
    "d": "right",
    "q": "search",
}
