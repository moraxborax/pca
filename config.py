from pathlib import Path

IMAGE_SIZE = 64
N_COMPONENTS = 128
SAMPLE_RATE_HZ = 10
N_FEATURES = IMAGE_SIZE * IMAGE_SIZE

CLASSES = ["forward", "backward", "stop", "left", "right", "search"]
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

DATA_DIR = Path("data")
ARTIFACTS_DIR = Path("artifacts")

RESIDUAL_THRESHOLD_MULTIPLIER = 2.0  # force search if residual > multiplier * train p95

KEY_TO_STATE = {
    "w": "forward",
    "s": "backward",
    "q": "stop",
    "a": "left",
    "d": "right",
    "e": "search",
}

