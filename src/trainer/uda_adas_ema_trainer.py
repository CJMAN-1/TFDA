import gc
import os
from typing import Tuple, Any

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader
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
from src.utils.optimizers import get_optimizer
from itertools import chain
import copy


class UDA_adas_trainer(Base_trainer):
    def __init__(self, config):
        super(UDA_adas_trainer, self).__init__()
        self.config = config
        ### Initialize etc variables
        self.best_miou = 0
        self.datasets = [self.config.source_data, self.config.target_data]
        self.source = self.config.source_data
        self.target = self.config.target_data
        self.class_num = self.config.class_num
        self.seg_loss_weight = self.config.seg_loss_weight

        self.loss_increment = float((self.config.target_seg_loss_weight[1] - self.config.seg_loss_weight[1]) / self.config.target_seg_loss_iteration)

        ### Logger
        self.LOG = utils.get_logger(__name__)

        ### Set seed for reproduction
        if config.get("rand_seed"):
            utils.set_seed(config.rand_seed)

        ### Initialize model
            ### segmentation model
        if config.get("seg_pretrained") and os.path.isfile(config.seg_pretrained):
            self.LOG.info(f"Load a trained segmentation model from {config.seg_pretrained}.")
            self.seg_model, self.seg_loss_set, self.seg_optimizer = self.load_seg_model(
                config.seg_pretrained, config.seg_model)
        else:
            self.LOG.info("There is no trained segmentation model, Initialize a new model.")
            self.seg_model, self.seg_loss_set, self.seg_optimizer = self.init_seg_model(config.seg_model)
        
            ### EMA teacher
        self.seg_ema_model = copy.deepcopy(self.seg_model)

            ### I2I model
        if config.get("i2i_pretrained") and os.path.isfile(config.i2i_pretrained):
            self.LOG.info(f"Load a trained I2I model from {config.i2i_pretrained}.")
            self.i2i_model = self.load_i2i_model(
                config.i2i_pretrained, config.i2i_model)
        else:
            self.LOG.info("There is no trained I2I model, Initialize a new model.")
            self.i2i_model = self.init_i2i_model(config.i2i_model)

            ### label filtering model
        self.label_filter = self.init_label_filter_model(config.label_filtering)

        ### Initialize dataloaders
        self.origin_loader = {}
        self.origin_loader['S_t'], self.origin_loader['T_t'], self.origin_loader['T_v'] = self.init_data_loader(config)
        
        self.loader = {}
        for k, v in self.origin_loader.items():
            self.loader[k] = iter(v)

        ### Initialize a tensorboard
        self.writer = hydra.utils.instantiate(config.logger)
        self.writer.add_text("description", self.config.description, 0)

        self.valid_class = self.origin_loader['T_v'].dataset.validclass_name


    def train(self) -> None:  
        self.LOG.info("All ranks are ready to train.")
        if 1:
            self.LOG.info("source only performance.")
            _, iou = self.eval(self.config, self.seg_model, self.origin_loader['T_v'])
            self.log_performance(iou, self.valid_class)
            self.plot_tensor_perform(iou, self.valid_class, 0)

        for iteration in tqdm(range(1, self.config.max_iteration), desc=self.config.ex+'| Training'):
            self.seg_model.train()
            self.i2i_model.train()
            batch = self.get_batch(self.loader)

            ### I2I - Compute output & optimize
            imgs, labels = dict(), dict()
            imgs[self.source], labels[self.source] = batch['S_t']['img'], batch['S_t']['label']
            imgs[self.target], labels[self.target] = batch['T_t']['img'], batch['T_t']['label']

            loss_d, loss_g, loss_dict, d_recons, id_recon, cvt_imgs = self.i2i_model(imgs, labels, mode='train', return_imgs=True)

            ### label filtering - make pseudo label and features
            if type(self.label_filter).__name__ == 'BARS' and self.config.bars_stop_iter >= iteration:
                with torch.no_grad():
                    ### forward model
                    self.seg_ema_model.eval()
                    _, feat_s2t = self.seg_ema_model(cvt_imgs[f'{self.source}2{self.target}'], mode='feat')
                    output, feat_t = self.seg_ema_model(batch['T_t']['img'], mode='feat')
                    
                    ### make pseudo label
                    pd_label = {}
                    pd_label[f'T_GT'] = batch['T_t']['label'] # easy to compare with pseudo label

                    pd_label[f'before_filtering_{self.target}'] = F.interpolate(output, batch['T_t']['img'].size()[2:], mode='bilinear')
                    pd_label[f'before_filtering_{self.target}'] = torch.argmax(pd_label[f'before_filtering_{self.target}'], dim=1)
                    pd_label[f'before_filtering_{self.target}'] = pd_label[f'before_filtering_{self.target}']

                    ### label filtering - compute filtered label
                    if iteration == 1: # 처음엔 label 그대로 사용해서 업데이트
                        self.label_filter.update(feat_s2t, batch['S_t']['label'], f'{self.source}2{self.target}')
                        self.label_filter.update(feat_t, pd_label[f'before_filtering_{self.target}'], self.target)
                        
                    mask_s2t, mask_t = self.label_filter(feat_s2t, feat_t, pd_label[f'before_filtering_{self.target}'], batch['S_t']['label'], f'{self.source}2{self.target}')
                    filtered_s2t_label = copy.deepcopy(batch['S_t']['label'].detach())
                    filtered_s2t_label[mask_s2t != filtered_s2t_label] = self.origin_loader['S_t'].dataset.ignore_label
                    pd_label[f'after_filtering_{self.source}2{self.target}'] = filtered_s2t_label

                    filtered_t_label = copy.deepcopy(pd_label[f'before_filtering_{self.target}'])
                    filtered_t_label[mask_t != filtered_t_label] = self.origin_loader['S_t'].dataset.ignore_label
                    pd_label[f'after_filtering_{self.target}'] = filtered_t_label

                    if iteration != 1: # 다음부턴 filtering된 label을 사용해서 업데이트
                        self.label_filter.update(feat_s2t, filtered_s2t_label, f'{self.source}2{self.target}')
                        self.label_filter.update(feat_t, filtered_t_label, self.target)

            ### make pseudo label
            elif (self.label_filter is None) or self.config.bars_stop_iter <= iteration:
                pd_label = {}
                with torch.no_grad():
                    pd_label[f'T_GT'] = batch['T_t']['label'] # easy to compare with pseudo label
                    
                    self.seg_ema_model.eval()
                    output = self.seg_ema_model(batch['T_t']['img'], mode='infer')
                    pd_label[f'before_filtering_{self.target}'] = F.interpolate(output, batch['T_t']['img'].size()[2:], mode='bilinear')
                    pd_label[f'before_filtering_{self.target}'] = torch.argmax(pd_label[f'before_filtering_{self.target}'], dim=1)
                    pd_label[f'before_filtering_{self.target}'] = pd_label[f'before_filtering_{self.target}'].long()
            
            # ### rectification artifact ignore
            ignore_top = 40
            ignore_bottom = 100
            pd_label[f'before_filtering_{self.target}'][:, :ignore_top, :] = self.origin_loader['S_t'].dataset.ignore_label
            pd_label[f'before_filtering_{self.target}'][:, -ignore_bottom:, :] = self.origin_loader['S_t'].dataset.ignore_label

            ### seg - Compute output
            if self.config.seg_start_iter <= iteration:
                self.seg_model.train()
                ### learning cvt imgs
                output_s2t = self.seg_model(cvt_imgs[f'{self.source}2{self.target}'])
                output_s2t = F.interpolate(output_s2t, batch['S_t']['label'].size()[1:], mode='bilinear', align_corners=False)
                loss_seg = self.seg_loss_set.CrossEntropy2d(output_s2t, batch['S_t']['label']) * self.seg_loss_weight[0]
                
                ### learning target imgs
                if type(self.label_filter).__name__ == 'BARS':
                    if type(self.label_filter).__name__ == 'BARS' and self.config.bars_stop_iter >= iteration:
                        output_t = self.seg_model(batch['T_t']['img'])
                        output_t = F.interpolate(output_t, filtered_t_label.size()[1:], mode='bilinear', align_corners=False)
                        loss_seg += self.seg_loss_set.CrossEntropy2d(output_t, filtered_t_label) * self.seg_loss_weight[1]
                    else : 
                        output_t = self.seg_model(batch['T_t']['img'])
                        output_t = F.interpolate(output_t, pd_label[f'before_filtering_{self.target}'].size()[1:], mode='bilinear', align_corners=False)
                        loss_seg += self.seg_loss_set.CrossEntropy2d(output_t, pd_label[f'before_filtering_{self.target}']) * self.seg_loss_weight[1]

                ### seg - Compute gradient & optimizer step
                self.seg_model.zero_grad()
                loss_seg.backward()
                self.seg_optimizer.step()

                ### seg - EMA teacher update
                self.update_ema(iteration)

                ### seg - adjust loss weight 
                self.seg_loss_weight[0] -= self.loss_increment
                self.seg_loss_weight[1] += self.loss_increment

            ### Tensorboard
            if iteration % self.config.tensor_interval == 0:
                if self.config.seg_start_iter <= iteration:
                    output_s2t = F.interpolate(output_s2t, batch['S_t']['label'].size()[1:], mode='bilinear', align_corners=False)
                    prediction_s2t = torch.argmax(output_s2t, dim = 1)
                    self.plot_tensor_img(prediction_s2t, batch, d_recons, id_recon, cvt_imgs, pd_label, iteration)

                    loss_sum = dict()
                    loss_sum['D'] = loss_d
                    loss_sum['G'] = loss_g
                    loss_sum['Seg'] = loss_seg
                else:
                    prediction_s2t = None
                    self.plot_tensor_img(prediction_s2t, batch, d_recons, id_recon, cvt_imgs, pd_label, iteration)

                    loss_sum = dict()
                    loss_sum['D'] = loss_d
                    loss_sum['G'] = loss_g
                self.plot_tensor_loss(loss_sum, loss_dict, iteration)

            ### Evaluation
            if iteration % self.config.eval_interval == 0:
                self.LOG.info(f"iteration: {iteration}")
                miou, iou = self.eval(self.config, self.seg_model, self.origin_loader['T_v'])
                self.log_performance(iou, self.valid_class)
                self.plot_tensor_perform(iou, self.valid_class, iteration)

                if miou > self.best_miou:
                    self.best_miou = miou
                    self.LOG.info(f'best miou : {self.best_miou:.2f} | miou : {miou:.2f}')
                    torch.save(self.seg_model.state_dict(), os.path.join(HydraConfig.get().run.dir, f'{type(self.seg_model).__name__}.pth'))
                    torch.save(self.i2i_model.state_dict(), os.path.join(HydraConfig.get().run.dir, f'{type(self.i2i_model).__name__}.pth'))
                gc.collect()
                torch.cuda.empty_cache()

            # gc.collect()
            # torch.cuda.empty_cache()

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

            batch[type]['img'] = batch[type]['img'].cuda()
            batch[type]['label'] = batch[type]['label'].cuda()

        return batch


    def load_seg_model(self, checkpoint_file: str, config: DictConfig) -> Tuple[Any, Any, Any]:
        # model
        model, loss_set, optimizer = self.init_seg_model(config)

        model.load_state_dict(
            torch.load(checkpoint_file)
        )
        return model, loss_set, optimizer


    def init_seg_model(self, config) -> Tuple[Any, Any, Any]:
        # model
        model = hydra.utils.instantiate(config.architecture).cuda()

        # criterion
        loss_set = Base_losses(self.class_num)
        
        # optimizer
        optimizer = get_optimizer(config.optimizer, model.parameters())
        return model, loss_set, optimizer


    def load_i2i_model(self, checkpoint_file: str, config: DictConfig) -> Tuple[Any, Any, Any]:
        # model
        model = self.init_i2i_model(config)

        model.load_state_dict(
            torch.load(checkpoint_file)
        )

        return model, 


    def init_i2i_model(self, config) -> Tuple[Any, Any, Any]:
        ### model
        model = hydra.utils.instantiate(config.model).cuda()
        
        return model

    def init_label_filter_model(self, config):
        ### BARS instantiate
        label_filter = None
        if config is not None:
            label_filter = hydra.utils.instantiate(config)

        return label_filter


    def init_data_loader(self, config: DictConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
        src_train_dataset = hydra.utils.instantiate(config.source_dataset.train)
        tar_train_dataset = hydra.utils.instantiate(config.target_dataset.train)
        tar_val_dataset =   hydra.utils.instantiate(config.target_dataset.val)
        

        src_train_dataloader = DataLoader(
                                            src_train_dataset,
                                            batch_size=config.data_loader.train.batch_size,
                                            pin_memory=config.data_loader.train.pin_memory,
                                            persistent_workers=config.data_loader.train.persistent_workers, # 한바퀴 돌고나서 메모리에서 안지우고 다시 쓰겠다.
                                            num_workers=config.data_loader.train.num_workers, # 사용할 cpu 코어 갯수
                                            prefetch_factor=config.data_loader.train.prefetch_factor,
                                            drop_last=True,
        )

        tar_train_dataloader = DataLoader(
                                            tar_train_dataset,
                                            batch_size=config.data_loader.train.batch_size,
                                            pin_memory=config.data_loader.train.pin_memory,
                                            persistent_workers=config.data_loader.train.persistent_workers,
                                            num_workers=config.data_loader.train.num_workers,
                                            prefetch_factor=config.data_loader.train.prefetch_factor,
                                            drop_last=True,
        )
        tar_val_dataloader = DataLoader(
                                            tar_val_dataset,
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
        iteration = 0
        with torch.no_grad():
            with logging_redirect_tqdm():
                for img, label in tqdm(dataloader):
                    iteration += 1
                    # Compute output
                    img, label = img.cuda(), label.cuda()
                    output = model(img, 'infer')
                    output = F.interpolate(output, label.size()[1:], mode='bilinear', align_corners=False)
                    output = F.log_softmax(output, dim = 1)
                    output = torch.argmax(output, dim = 1)
                    
                    # miou
                    conf_mat += metric.conf_mat(label.cpu().numpy(), output.cpu().numpy(), config.class_num)
            
        iou = metric.iou(conf_mat)
        miou = np.nanmean(iou)
        
        return miou, iou

    def inference(self, config, model, dataloader):
        model.eval()
        conf_mat = np.zeros((config.class_num,) * 2)
        miou = np.zeros(shape=1)
        iou = np.zeros(shape=config.class_num)
        iteration = 0
        with torch.no_grad():
            with logging_redirect_tqdm():
                for img, label in tqdm(dataloader):
                    iteration += 1
                    # Compute output
                    pd_label = {}
                    img, label = img.cuda(), label.cuda()
                    output = model(img, 'infer')
                    output = F.interpolate(output, label.size()[1:], mode='bilinear', align_corners=False)
                    output = F.log_softmax(output, dim = 1)
                    pd_label['before_filtering'] = torch.argmax(output, dim = 1)
                    
                    # miou
                    conf_mat += metric.conf_mat(label.cpu().numpy(), pd_label['before_filtering'].cpu().numpy(), config.class_num)

                    ### thresholding
                    pd_label['after_filtering'] = copy.deepcopy(pd_label['before_filtering'])
                    pd_label['after_filtering'][torch.argmax(output, dim=1) < 0.5] = -1
                    
                    batch = {}
                    data = {}
                    data['img'], data['label'] = img, label
                    batch['T_v'] = data

                    self.plot_tensor_img(batch = batch, pd_label=pd_label, iteration=iteration)
            
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
        
        
    def plot_tensor_img(self, prediction_s2t=None, batch=None, d_recons=None, id_recon=None, cvt_imgs=None, pd_label=None, iteration=0):
        with torch.no_grad():
            ### input
            if batch is not None:
                for domain, val in batch.items():
                    img_grid = torchvision.utils.make_grid(val['img'], normalize=True, value_range=(-1,1))
                    self.writer.add_image(f'input/{domain[0]}_img', img_grid, iteration)
                    lbl = self.origin_loader['S_t'].dataset.colorize_label(val['label'])
                    img_grid = torchvision.utils.make_grid(lbl, normalize=True, value_range=(0,255))
                    self.writer.add_image(f'input/{domain[0]}_GT', img_grid, iteration)
                
                # ### target input
                # img_grid = torchvision.utils.make_grid(batch['T_t']['img'], normalize=True, value_range=(-1,1))
                # self.writer.add_image('input/target_img', img_grid, iteration)
                # lbl = self.origin_loader['S_t'].dataset.colorize_label(batch['T_t']['label'])
                # img_grid = torchvision.utils.make_grid(lbl, normalize=True, value_range=(0,255))
                # self.writer.add_image('input/target_GT', img_grid, iteration)

            ### output
            if prediction_s2t is not None:
                prediction_s2t = self.origin_loader['S_t'].dataset.colorize_label(prediction_s2t)
                img_grid = torchvision.utils.make_grid(prediction_s2t, normalize=True, value_range=(0,255))
                self.writer.add_image('output/prediction_s2t', img_grid, iteration)

            ### direct recon
            if d_recons is not None:
                for name, img in d_recons.items():
                    img_grid = torchvision.utils.make_grid(img, normalize=True, value_range=(-1,1))
                    self.writer.add_image(f'direct recon/{name}', img_grid, iteration)
            ### indirect recon
            if id_recon is not None:
                for name, img in id_recon.items():
                    img_grid = torchvision.utils.make_grid(img, normalize=True, value_range=(-1,1))
                    self.writer.add_image(f'indirect recon/{name}', img_grid, iteration)

            ### converted image
            if cvt_imgs is not None:
                for name, img in cvt_imgs.items():
                    img_grid = torchvision.utils.make_grid(img, normalize=True, value_range=(-1,1))
                    self.writer.add_image(f'converted image/{name}', img_grid, iteration)

            ### pseudo label
            if pd_label is not None:
                for name, img in pd_label.items():
                    img = self.origin_loader['S_t'].dataset.colorize_label(img)
                    img_grid = torchvision.utils.make_grid(img, normalize=True, value_range=(0,255))
                    self.writer.add_image(f'pseudo_label/{name}', img_grid, iteration)

            ### waiting, 이미지가 저장되는데 시간이 어느정도 필요함.
            sleep(0.5)


    def plot_tensor_perform(self, iou, class_name, iteration):
        with torch.no_grad():
            class_name = class_name
            for name, value in zip(class_name, iou):
                self.writer.add_scalar(f'iou/{name}', value, iteration)
            miou = np.nanmean(iou)
            self.writer.add_scalar(f'iou/0.mIoU', miou, iteration)


    def plot_tensor_loss(self, loss_sum, loss_dict, iteration):
        with torch.no_grad():
            for name, value in loss_sum.items():
                self.writer.add_scalar(f'loss/{name}', value, iteration)

            for name, value in loss_dict.items():
                self.writer.add_scalar(f'loss/{name}', value, iteration)

    def update_ema(self, iteration):
        alpha_teacher = min(1 - 1/ (iteration), self.config.ema_alpha)

        for ema_param, param in zip(self.seg_ema_model.parameters(), self.seg_model.parameters()):
            if not param.data.shape:
                ema_param.data = alpha_teacher * ema_param.data + (1-alpha_teacher) * param.data

            else:
                ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1-alpha_teacher) * param[:].data[:]