import hydra
from omegaconf import DictConfig
import os
import sys
from src.utils import get_logger
import torch, gc

@hydra.main(config_path="configs", config_name="config_mtdtnet")
def main(config: DictConfig):
    os.chdir(config.work_dir)
    ### your code ###
    from src.trainer.uda_mtdt_trainer import UDA_mtdt_trainer
    trainer = UDA_mtdt_trainer(config)
    ### your code ###
    
    return trainer.train()


if __name__ == "__main__":
    main()
    gc.collect()
    torch.cuda.empty_cache()
