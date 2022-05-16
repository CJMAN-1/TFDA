import hydra
from omegaconf import DictConfig
import os

@hydra.main(config_path="configs", config_name="config")
def main(config: DictConfig):
    from src.train import train
    os.chdir(config.work_dir)
    return train(config)


if __name__ == "__main__":
    main()
