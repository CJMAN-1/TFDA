import torch
import os

class Base_trainer():
    def __init__(self):
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        self.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(self.local_rank)


    def __del__(self):
        torch.distributed.destroy_process_group()