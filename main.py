import numpy as np

from pca.pca import pca, pca_transform

from fastapi import FastAPI

import torch
import torch.nn as nn
from train import Net, train
from inference import inference, ACTIONS

device = "cpu"

app = FastAPI()










