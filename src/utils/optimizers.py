import torch.optim
from src.utils import get_logger

def get_optimizer(config, parameter):
    LOG = get_logger(__name__)

    if config.type == 'Adam':
        optimizer = torch.optim.Adam(
            params = parameter,
            lr = config.lr,
            betas = tuple(config.betas),
            weight_decay = config.weight_decay
        )
    elif config.type == 'AdamW':
        optimizer = torch.optim.AdamW(
            params = parameter,
            lr = config.lr,
            betas = tuple(config.betas),
            weight_decay = config.weight_decay
        )
    else:
        LOG.info(f"There is no [{config.type}] optimizer.")
        raise ValueError

    return optimizer