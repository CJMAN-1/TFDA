import os
import torch
import numpy as np
import random
import logging
from typing import Callable, Any
from omegaconf import DictConfig

# --------- decorator --------- #


def set_seed(random_seed: int) -> None:
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def get_logger(name=__name__) -> logging.Logger:
    logger = logging.getLogger(name)

    return logger
