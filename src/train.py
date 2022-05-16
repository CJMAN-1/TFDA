import gc
import os
from typing import Tuple, Any

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from src import utils

LOG = utils.get_logger(__name__)


@utils.set_dist_func # 병렬로 돌아가게 해주는 데코레이터
def train(config: DictConfig, local_rank: int) -> None: # local_rank : 한개의 머신(gpu가 달린 컴퓨터) 안에 있는 process의 인덱스, global_rank도 있는데 그건 머신이 여러개 있을 경우에 의미가 있음.
    # Set seed for reproduction
    if config.get("rand_seed"):
        utils.set_seed(config.rand_seed)

    # Initialize model
    if config.get("checkpoint_file") and os.path.isfile(config.checkpoint_file):
        LOG.info("Load a trained model.")
        model, criterion, optimizer = load_model(
            config.checkpoint_file, config.model, local_rank
        )
    else:
        LOG.info("There is no trained model, Initialize a new model.")
        model, criterion, optimizer = init_model(config.model, local_rank)
    
    # Initialize dataloaders
    origin_loader = {}
    origin_loader['S_t'], origin_loader['T_t'], origin_loader['T_v'] = init_data_loader(config)
    

    loader = {}
    for k, v in origin_loader.items():
        loader[k] = iter(v)

    # Initialize a tensorboard (only zero rank)
    writer = hydra.utils.instantiate(config.logger) if local_rank == 0 else None # log는 local rank가 0인 애로만 작성. 이래도 아무 상관 없나?

    dist.barrier()
    LOG.info("All ranks are ready to train.")
 
    for iteration in tqdm(range(1, config.max_iteration+1), desc=config.ex+'| Training'):
        batch = get_batch(loader, origin_loader, local_rank)
        
        # Compute output # TODO : segformer 모델 포팅
        outputs = model(batch)
        loss = criterion(outputs, label)

        # Compute gradient & optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # TODO : add tensorboard
        
        if iteration % config.eval_freq == 0:
            model.eval()
            with torch.no_grad():
                for img, label in tqdm(origin_loader['T_v']):
                    batch = {}
                    batch['T_v']['img'] = img.to(local_rank)
                    label['T_v']['label'] = label.to(local_rank)

                    # Compute output
                    outputs = model(batch)
            

        gc.collect()
        torch.cuda.empty_cache()

    if writer:
        writer.close()
    if local_rank == 0:
        torch.save(model.state_dict(), os.path.join(os.getcwd(), "model.pt"))
        LOG.info("Saved model.")

    LOG.info("Finished train.")


def get_batch(loader, origin_loader, local_rank): # -> img : B3HW, label : BHW
    batch = dict()
    data = dict()
    for type in ['S_t', 'T_t']:
        try:
            data['img'], data['label'] = next(loader[type])
            batch[type] = data.copy()
        except StopIteration:
            loader[type] = iter(origin_loader[type])
            data['img'], data['label'] = next(loader[type])
            batch[type] = data.copy()

        batch[type]['img'] = batch[type]['img'].to(local_rank)
        batch[type]['label'] = batch[type]['label'].to(local_rank)

    return batch

def load_model(
    checkpoint_file: str, config: DictConfig, local_rank: int
) -> Tuple[Any, Any, Any]:
    # model
    model, criterion, optimizer = init_model(config, local_rank)

    model.load_state_dict(
        torch.load(checkpoint_file, map_location=f"cuda:{local_rank}")
    )

    return model, criterion, optimizer


def init_model(config: DictConfig, local_rank: int) -> Tuple[Any, Any, Any]:
    # model
    model = hydra.utils.instantiate(config.architecture).to(local_rank)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # criterion
    criterion = torch.nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay,
    )
    return model, criterion, optimizer


def init_data_loader(config: DictConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    src_train_dataset = hydra.utils.instantiate(config.source_dataset.train)
    tar_train_dataset = hydra.utils.instantiate(config.target_dataset.train)
    tar_val_dataset =   hydra.utils.instantiate(config.target_dataset.val)
    
    src_train_sampler = DistributedSampler(src_train_dataset, shuffle=True, drop_last=True)
    tar_train_sampler = DistributedSampler(tar_train_dataset, shuffle=True, drop_last=True)
    tar_val_sampler =   DistributedSampler(tar_val_dataset)

    src_train_dataloader = DataLoader(
                                        src_train_dataset,
                                        sampler=src_train_sampler,
                                        batch_size=config.data_loader.train.batch_size,
                                        pin_memory=config.data_loader.train.pin_memory,
                                        persistent_workers=config.data_loader.train.persistent_workers, # 한바퀴 돌고나서 메모리에서 안지우고 다시 쓰겠다.
                                        num_workers=config.data_loader.train.num_workers, # 사용할 cpu 코어 갯수
                                        prefetch_factor=config.data_loader.train.prefetch_factor,
                                        drop_last=True,
    )

    tar_train_dataloader = DataLoader(
                                        tar_train_dataset,
                                        sampler=tar_train_sampler,
                                        batch_size=config.data_loader.train.batch_size,
                                        pin_memory=config.data_loader.train.pin_memory,
                                        persistent_workers=config.data_loader.train.persistent_workers,
                                        num_workers=config.data_loader.train.num_workers,
                                        prefetch_factor=config.data_loader.train.prefetch_factor,
                                        drop_last=True,
    )

    tar_val_dataloader = DataLoader(
                                        tar_val_dataset,
                                        sampler=tar_val_sampler,
                                        batch_size=config.data_loader.val.batch_size,
                                        persistent_workers=config.data_loader.train.persistent_workers,
                                        num_workers=config.data_loader.train.num_workers,
    )

    return src_train_dataloader, tar_train_dataloader, tar_val_dataloader
