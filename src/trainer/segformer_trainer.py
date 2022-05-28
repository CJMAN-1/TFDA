import gc
import os
from typing import Tuple, Any

import hydra
import torch
import torch.nn.functional as F
import torch.distributed as dist
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from src import utils
from src.utils.losses import *
from src.utils import metric
import numpy as np
from hydra.core.hydra_config import HydraConfig
from tqdm.contrib.logging import logging_redirect_tqdm
from prettytable import PrettyTable
import torchvision
from time import sleep
from src.trainer.base_trainer import Base_trainer

class Seg_trainer(Base_trainer):
    def __init__(self, config):
        super(Seg_trainer, self).__init__(config)
        self.config = config
        ### Logger
        self.LOG = utils.get_logger(__name__)

        ### Set seed for reproduction
        if config.get("rand_seed"):
            utils.set_seed(config.rand_seed)

        ### Initialize model
        if config.get("seg_pretrained") and os.path.isfile(config.seg_pretrained):
            self.LOG.info("Load a trained model.")
            self.model, self.criterion, self.optimizer = self.load_model(
                config.seg_pretrained, config.seg_model, self.local_rank)
        else:
            self.LOG.info("There is no trained segmentor model, Initialize a new model.")
            self.model, self.criterion, self.optimizer = self.init_model(config.seg_model)

        ### Initialize dataloaders
        self.origin_loader = {}
        self.origin_loader['S_t'], self.origin_loader['T_t'], self.origin_loader['T_v'] = self.init_data_loader(config)
        

        self.loader = {}
        for k, v in self.origin_loader.items():
            self.loader[k] = iter(v)

        ### Initialize a tensorboard (only zero rank)
        self.writer = hydra.utils.instantiate(config.logger) if self.local_rank == 0 else None # log는 local rank가 0인 애로만 작성. 이래도 아무 상관 없나?
        self.valid_class = self.origin_loader['T_v'].dataset.validclass_name

        ### Initialize etc variables
        self.best_miou = 0
        # if self.local_rank == 0: => 이거 했는데 다 똑같은 색깔로만 나옴.. 개열받음
        #     iou_class = ['iou/'+name for name in self.valid_class]
        #     layout = {'Common values': {'iou':['Multiline', iou_class]}}
        #     self.writer.add_custom_scalars(layout)


    def __del__(self):
        super(Seg_trainer, self).__del__()

        if self.writer:
            self.writer.close()


    def train(self) -> None: # self.local_rank : 한개의 머신(gpu가 달린 컴퓨터) 안에 있는 process의 인덱스, global_rank도 있는데 그건 머신이 여러개 있을 경우에 의미가 있음.   
        dist.barrier()
        self.LOG.info("All ranks are ready to train.")
        for iteration in tqdm(range(0, self.config.max_iteration), desc=self.config.ex+'| Training'):
            self.model.train()
            batch = self.get_batch(self.loader)
            
            ### Compute output
            output = self.model(batch['S_t']['img'])
            output = F.interpolate(output, batch['S_t']['label'].size()[1:], mode='bilinear', align_corners=False)
            loss = self.criterion(output, batch['S_t']['label'])
            
            ### Compute gradient & optimizer step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            ### Tensorboard
            if iteration % self.config.tensor_interval == 0 and self.local_rank == 0:
                output = F.log_softmax(output, dim = 1)
                prediction = torch.argmax(output, dim = 1)
                self.plot_tensor_img(prediction, batch, iteration)

            ### Evaluation
            if iteration % self.config.eval_interval == 0 and self.local_rank == 0:
                self.LOG.info(f"iteration: {iteration}")
                miou, iou = self.eval(self.config, self.model, self.origin_loader['T_v'])
                self.log_performance(iou, self.valid_class)
                self.plot_tensor_perform(iou, self.valid_class, iteration)

                if miou > self.best_miou:
                    self.best_miou = miou
                    self.LOG.info(f'best miou : {self.best_miou:.2f} | miou : {miou:.2f}')
                    torch.save(self.model.state_dict(), os.path.join(HydraConfig.get().run.dir, f'{type(self.model.module).__name__}.pth'))

                gc.collect()
                torch.cuda.empty_cache()

        self.LOG.info("Finish training.")


    def get_batch(self, loader): # -> img : B3HW, label : BHW
        batch = dict()
        data = dict()
        for type in ['S_t', 'T_t']:
            try:
                data['img'], data['label'] = next(loader[type])
                batch[type] = data.copy()
            except StopIteration:
                loader[type] = iter(self.origin_loader[type])
                data['img'], data['label'] = next(loader[type])
                batch[type] = data.copy()

            batch[type]['img'] = batch[type]['img'].to(self.local_rank)
            batch[type]['label'] = batch[type]['label'].to(self.local_rank)

        return batch


    def load_model(self, checkpoint_file: str, config: DictConfig) -> Tuple[Any, Any, Any]:
        # model
        model, criterion, optimizer = self.init_model(config, self.local_rank)

        model.load_state_dict(
            torch.load(checkpoint_file, map_location=f"cuda:{self.local_rank}")
        )

        return model, criterion, optimizer


    def init_model(self, config) -> Tuple[Any, Any, Any]:
        # model
        model = hydra.utils.instantiate(config.architecture).to(self.local_rank)
        model = DDP(model, device_ids=[self.local_rank], find_unused_parameters=False)

        # criterion
        loss_set = Base_losses()
        criterion = getattr(loss_set, config.loss.type)
        
        # optimizer
        if config.optimizer.type == 'AdamW':
            optimizer = torch.optim.AdamW(
                params = model.parameters(),
                lr = config.optimizer.lr,
                betas = tuple(config.optimizer.betas),
                weight_decay = config.optimizer.weight_decay
            )
        return model, criterion, optimizer


    def init_data_loader(self, config: DictConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
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
                                            drop_last=False,
        )

        return src_train_dataloader, tar_train_dataloader, tar_val_dataloader


    def eval(self, config, model, dataloader):
        model.eval()
        conf_mat = np.zeros((config.class_num,) * 2)
        miou = np.zeros(shape=1)
        iou = np.zeros(shape=config.class_num)

        with torch.no_grad():
            with logging_redirect_tqdm():
                for img, label in tqdm(dataloader):
                    # Compute output
                    output = model(img)
                    output = F.interpolate(output, label.size()[1:], mode='bilinear', align_corners=False)
                    output = F.log_softmax(output, dim = 1)
                    output = torch.argmax(output, dim = 1)
                    # miou
                    conf_mat += metric.conf_mat(label.cpu().numpy(), output.cpu().numpy(), config.class_num)
                
            iou = metric.iou(conf_mat)
            miou = np.nanmean(iou)
            
            return miou, iou


    def log_performance(self, iou, class_name):
        class_name = class_name + ['mIoU']
        iou = np.append(iou, np.nanmean(iou))
        iou = np.round(iou, 2)
        remainder = 1 if len(class_name) % 10 > 0 else 0
        rows = int(len(class_name) / 10) + remainder
        
        for i in range(0, rows):
            table = PrettyTable()
            table.field_names = class_name[10*i : 10*(i+1)]
            table.add_row(iou[10*i : 10*(i+1)])
            self.LOG.info('\n'+table.get_string())
        
        
    def plot_tensor_img(self, prediction, batch, iteration):
        with torch.no_grad():
            ### input
            img_grid = torchvision.utils.make_grid(batch['S_t']['img'], normalize=True, value_range=(-1,1))
            self.writer.add_image('input/img', img_grid, iteration)
            lbl = self.origin_loader['S_t'].dataset.colorize_label(batch['S_t']['label'])
            img_grid = torchvision.utils.make_grid(lbl, normalize=True, value_range=(0,255))
            self.writer.add_image('input/GT', img_grid, iteration)

            ### output
            prediction = self.origin_loader['S_t'].dataset.colorize_label(prediction)
            img_grid = torchvision.utils.make_grid(prediction, normalize=True, value_range=(0,255))
            self.writer.add_image('output/prediction', img_grid, iteration)

            ### waiting, 이미지가 저장되는데 시간이 어느정도 필요함.
            sleep(0.5)

    def plot_tensor_perform(self, iou, class_name, iteration):
        with torch.no_grad():
            for name, value in zip(class_name, iou):
                self.writer.add_scalar(f'iou/{name}', value, iteration)
        