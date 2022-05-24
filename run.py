import hydra
from omegaconf import DictConfig
import os
import sys
from src.utils import get_logger

@hydra.main(config_path="configs", config_name="config")
def main(config: DictConfig):
    os.chdir(config.work_dir)
    ### your code ###
    from src.trainer.seg_trainer import Seg_trainer
    LOG = get_logger(__name__)
    LOG.info(sys.argv)
    trainer = Seg_trainer(config)
    ### your code ###
    
    return trainer.train()


if __name__ == "__main__":
    main()
