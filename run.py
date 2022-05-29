import hydra
from omegaconf import DictConfig
import os
import sys
from src.utils import get_logger
import torch, gc

@hydra.main(config_path="configs", config_name="config_adas")
def main(config: DictConfig):
    os.chdir(config.work_dir)
    os.environ['HYDRA_FULL_ERROR'] = '1'
    ### your code ###
    from src.trainer.uda_adas_trainer import UDA_adas_trainer
    trainer = UDA_adas_trainer(config)
    ### your code ###
    
    return trainer.train()


if __name__ == "__main__":
    main()