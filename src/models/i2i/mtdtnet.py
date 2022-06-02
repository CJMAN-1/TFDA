import torch.nn as nn
import functools
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
import torch
from src.utils.losses import *
from src.utils import get_logger
import os
from itertools import chain
from src.utils.optimizers import get_optimizer


class Mtdtnet(nn.Module):
    def __init__(self,
                architecture,
                loss,
                generator_optimizer,
                discriminator_optimizer,
                datasets,  # 0: source 1~: targets
                pretrained_mtdtnet,
                class_num):
        super(Mtdtnet, self).__init__()
        self.encoder = architecture.encoder
        self.generator = architecture.generator
        self.st_encoder = architecture.st_encoder
        self.discrminator = architecture.discrminator
        self.domain_transfer = architecture.domain_transfer
        self.label_embed = architecture.label_embed
        self.datasets = datasets
        self.source = self.datasets[0]
        self.targets = self.datasets[1:]
        self.converts = [self.source+'2'+target for target in self.targets]
        self.class_num = class_num

        self.init_weights(pretrained_mtdtnet)

        ### logger
        self.LOG = get_logger(__name__)

        ### criterion
        self.loss_set = Mtdt_losses(self.datasets, self.class_num)
        self.loss_weights = {}

        self.LOG.info(f'losses of I2I model: {loss.type}')
        for name, w in zip(loss.type, loss.weight):
            self.loss_weights[name] = w

        ### optimizer
        param_g = self.encoder.parameters()
        param_g = chain(param_g, self.generator.parameters())
        param_g = chain(param_g, self.st_encoder.parameters())
        param_g = chain(param_g, self.label_embed.parameters())
        param_g = chain(param_g, self.domain_transfer.parameters())

        pram_d = self.discrminator.parameters()
        self.optimizers = {}
        self.optimizers['G'] = get_optimizer(generator_optimizer, param_g)
        self.optimizers['D'] = get_optimizer(discriminator_optimizer, pram_d)

        self.is_first_step = True
        
    
    def init_weights(self, pretrained=None):
        
        if pretrained is not None:
            self.LOG.info(f'load mtdtnet from: {pretrained}')
            self.load_state_dict(torch.load(pretrained))

    def _forward_train(self, imgs, labels, return_imgs=True):
        if self.is_first_step:
            self.is_first_step = False
            ### train generator
            loss_g, loss_dict_g, d_recons, id_recon, cvt_imgs = self._train_gen(imgs, labels)
            
            loss_g.backward()
            self.optimizers['G'].step()
            self.optimizers['G'].zero_grad()

            ### train discriminator
            loss_d, loss_dict_d = self._train_dis(imgs, labels)

            loss_d.backward()
            self.optimizers['D'].step()
            self.optimizers['D'].zero_grad()
        else:
            ### train discriminator
            loss_d, loss_dict_d = self._train_dis(imgs, labels)

            loss_d.backward()
            self.optimizers['D'].step()
            self.optimizers['D'].zero_grad()

            ### train generator
            loss_g, loss_dict_g, d_recons, id_recon, cvt_imgs = self._train_gen(imgs, labels)
            
            loss_g.backward()
            self.optimizers['G'].step()
            self.optimizers['G'].zero_grad()

        loss_dict = dict(loss_dict_g, **loss_dict_d)

        ### return        
        if return_imgs:
            for k, img in d_recons.items():
                d_recons[k] = img.detach()
            for k, img in id_recon.items():
                id_recon[k] = img.detach()
            for k, img in cvt_imgs.items():
                cvt_imgs[k] = img.detach()
            return loss_d, loss_g, loss_dict, d_recons, id_recon, cvt_imgs
        else:
            return loss_d, loss_g, loss_dict,

        
    def _forward_infer(self, imgs, labels):
        raise NotImplementedError


    def forward(self, imgs, labels, mode=None, return_imgs=True):
        '''
        imgs : dict()
        labels : dict()
        mode : 'train' or 'infer'
        return_imgs : boolean. return images(direct recon, indirect recon, cvt_imgs) during training
        '''

        if mode == 'train' or mode is None:
            return self._forward_train(imgs, labels, return_imgs)
        elif mode == 'infer':
            return self._forward_infer(imgs, labels)


    def _train_dis(self, imgs, labels):
        ### init variables
        loss_dis = 0
        loss = dict()
        D_outputs_real, D_outputs_fake = dict(), dict()
        converted_imgs = dict()
        gamma, beta = dict(), dict()
        
        ### set disciminator requires_grd to True
        self.discrminator.requires_grad_(True)
        self.encoder.requires_grad_(False)
        self.generator.requires_grad_(False)
        self.st_encoder.requires_grad_(False)
        self.label_embed.requires_grad_(False)
        self.domain_transfer.requires_grad_(False)
        
        self.discrminator.zero_grad()
        

        ### forward real target images
        for dset in self.datasets:
            if dset in self.targets:
                D_outputs_real[dset] = self.discrminator(self._slice_patches(imgs[dset]))

        ### forward fake(converted source -> target) images
        gamma[self.source], beta[self.source] = self.st_encoder(imgs[self.source])
        for convert in self.converts:
            with torch.no_grad():
                source, target = convert.split('2')
                gamma[convert], beta[convert] = self.domain_transfer(gamma[source], beta[source], target=target)
                converted_imgs[convert] = self.generator(gamma[convert]*self.label_embed(labels[source]) + beta[convert])
            D_outputs_fake[convert] = self.discrminator(self._slice_patches(converted_imgs[convert]))
            
        loss['Gan_d'] = self.loss_set.Gan_d(D_outputs_real, D_outputs_fake, self.targets)
        loss_dis = loss['Gan_d'] * self.loss_weights['Gan_d']

        return loss_dis, loss


    def _train_gen(self, imgs, labels):
        ### init variables
        loss_gen = 0
        loss = {}
        d_recons, id_recon = dict(), dict()
        cvt_imgs = dict()
        gamma, beta = dict(), dict()
        features = dict()
        D_outputs_fake = dict()

        ### set discriminator requires_grad to false
        self.discrminator.requires_grad_(False)
        self.encoder.requires_grad_(True)
        self.generator.requires_grad_(True)
        self.st_encoder.requires_grad_(True)
        self.label_embed.requires_grad_(True)
        self.domain_transfer.requires_grad_(True)
        self.encoder.zero_grad()
        self.generator.zero_grad()
        self.st_encoder.zero_grad()
        self.label_embed.zero_grad()
        self.domain_transfer.zero_grad()

        ### direct recon (for all domains)
        for dset in self.datasets:
            features[dset] = self.encoder(imgs[dset])
            d_recons[dset] = self.generator(features[dset])
        
        ### indirect recon (only source domain)
        with torch.no_grad():
            for dset in self.datasets:
                features[dset] = self.encoder(imgs[dset])
                if dset in self.targets:
                    self.domain_transfer.update(features[dset], dset)
        gamma[self.source], beta[self.source] = self.st_encoder(imgs[self.source])
        id_recon[self.source] = self.generator(gamma[self.source]*self.label_embed(labels[self.source]) + beta[self.source])
        
        ### converted image & discriminator output (source -> targets)
        for convert in self.converts:
            source, target = convert.split('2')
            gamma[convert], beta[convert] = self.domain_transfer(gamma[source], beta[source], target=target)
            cvt_imgs[convert] = self.generator(gamma[convert]*self.label_embed(labels[source]) + beta[convert])
            D_outputs_fake[convert] = self.discrminator(self._slice_patches(cvt_imgs[convert]))
        
        ### compute loss
        loss['Gan_g'] = self.loss_set.Gan_g(D_outputs_fake)
        loss['Direct_recon'] = self.loss_set.Direct_recon(imgs, d_recons)
        loss['Indirect_recon'] = self.loss_set.Indirect_recon(imgs[self.source], id_recon[self.source])
        loss['Consis'] = self.loss_set.Consis(imgs[self.source], cvt_imgs)
        loss['Style'] = self.loss_set.Style(imgs, cvt_imgs)
        
        for k, v in loss.items():
            if k not in ['Gan_d']:
                loss_gen += v * self.loss_weights[k]
            
        return loss_gen, loss, d_recons, id_recon, cvt_imgs


    def _slice_patches(self, imgs, hight_slice=2, width_slice=4):
        b, c, h, w = imgs.size()
        h_patch, w_patch = int(h / hight_slice), int(w / width_slice)
        patches = imgs.unfold(2, h_patch, h_patch).unfold(3, w_patch, w_patch)
        patches = patches.contiguous().view(b, c, -1, h_patch, w_patch)
        patches = patches.transpose(1,2)
        patches = patches.reshape(-1, c, h_patch, w_patch)
        return patches

class Encoder(nn.Module):
    def __init__(self, channels=3):
        super(Encoder, self).__init__()
        bin = functools.partial(nn.GroupNorm, 4)
        # bin = functools.partial(Normlayer, affine=True)
        self.Encoder_Conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=4, stride=2, padding=1, bias=False),
            bin(32),
            nn.ReLU(True),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            bin(64),
            nn.ReLU(True),
        )

    def forward(self, inputs):
        # output = F.normalize(self.Encoder_Conv(inputs), dim=0) # batch_size x 512 x 10 x 6
        output = self.Encoder_Conv(inputs)
        return output


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        bin = functools.partial(nn.GroupNorm, 4)
        self.Decoder_Conv = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True)),
            bin(32),
            nn.ReLU(True),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            # batch_size x 3 x 1280 x 768
            spectral_norm(nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, bias=True)),            
            nn.Tanh()
        )
    def forward(self, x):
        return self.Decoder_Conv(x)


class Style_Encoder(nn.Module):
    def __init__(self, channels=3):
        super(Style_Encoder, self).__init__()
        bin = functools.partial(nn.GroupNorm, 4)
        # bin = functools.partial(Normlayer, affine=True)
        self.Encoder_Conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=4, stride=2, padding=1, bias=False),
            bin(32),
            nn.ReLU(True),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
        )
        self.gamma = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
        )
        self.beta = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
        )

    def forward(self, inputs):
        output = self.Encoder_Conv(inputs) # batch_size x 512 x 10 x 6
        # gamma = F.normalize(self.gamma(output), dim=0)
        # beta = F.normalize(self.beta(output), dim=0)
        gamma = self.gamma(output)
        beta = self.beta(output)
        return gamma, beta


class Multi_Head_Discriminator(nn.Module):
    def __init__(self, num_domains, channels=3):
        super(Multi_Head_Discriminator, self).__init__()
        self.Conv = nn.Sequential(
            # input size: 256x256
            spectral_norm(nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1, bias=True)),  # 
            # nn.InstanceNorm2d(64),
            nn.GroupNorm(4, 64),
            nn.LeakyReLU(negative_slope=0.2),
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True)),  # 
            nn.GroupNorm(4, 128),
            # nn.InstanceNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.Patch = nn.Sequential(
            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True)),  # 
            # nn.InstanceNorm2d(256),
            nn.GroupNorm(4, 256),
            nn.LeakyReLU(negative_slope=0.2),
            spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=True)), # 
            # nn.InstanceNorm2d(512),
            nn.GroupNorm(4, 512),
            nn.LeakyReLU(negative_slope=0.2),
            spectral_norm(nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1, bias=True)),
        )

        # self.fc = nn.Sequential(
        #     spectral_norm(nn.Linear(64*64*128, 500)),
        #     nn.ReLU(),
        #     spectral_norm(nn.Linear(500, num_domains))
        # )

    def forward(self, inputs):
        conv_output = self.Conv(inputs)
        patch_output = self.Patch(conv_output)
        # fc_output = self.fc(conv_output.view(conv_output.size(0), -1))
        # fc_output = self.fc(conv_output).view(conv_output.size(0), -1)
        # return (patch_output, fc_output)
        return patch_output


class Domain_Transfer(nn.Module):
    def __init__(self, targets):
        super().__init__()
        self.n = dict()
        self.m = dict()
        self.s = dict()
        self.w = 1

        for dset in targets:
            self.n[dset] = 0
            self.m[dset] = 0
            self.s[dset] = 0
        
        self.gamma_res1 = D_AdaIN_ResBlock(64, 64, targets)
        self.gamma_res2 = D_AdaIN_ResBlock(64, 64, targets)
        self.beta_res1 = D_AdaIN_ResBlock(64, 64, targets)
        self.beta_res2 = D_AdaIN_ResBlock(64, 64, targets)

    def forward(self, gamma, beta, target):
        # Domain mean, std
        target_mean = self.m[target].mean(dim=(0,2,3)).unsqueeze(0)
        target_std = ((self.s[target].mean(dim=(0,2,3)))/self.n[target]).sqrt()
        gamma_convert = self.gamma_res1(gamma, target_mean, target_std, target)
        gamma_convert = self.gamma_res2(gamma_convert, target_mean, target_std, target)
        beta_convert = self.gamma_res1(beta, target_mean, target_std, target)
        beta_convert = self.gamma_res2(beta_convert, target_mean, target_std, target)
        
        return gamma_convert, beta_convert
    
    def update(self, feature, target):
        self.n[target] += 1
        if self.n[target] == 1:
            self.m[target] = feature
            self.s[target] = (feature - self.m[target].mean(dim=(0,2,3), keepdim=True)) ** 2
        else:
            prev_m = self.m[target].mean(dim=(0,2,3), keepdim=True)  # 1 x C x 1 x 1
            self.m[target] += self.w * (feature - self.m[target]) / self.n[target]  # B x C x H x W
            curr_m = self.m[target].mean(dim=(0,2,3), keepdim=True)  # 1 x C x 1 x 1
            self.s[target] += self.w * (feature - prev_m) * (feature - curr_m)  # B x C x H x W


class Label_Embed(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Conv2d(1, 64, 1, 1, 0)
    
    def forward(self, seg):
        return self.embed(F.interpolate(seg.unsqueeze(1).float(), size=(256, 512), mode='nearest'))

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, filters=64, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        bin = functools.partial(nn.GroupNorm, 4)
        # bin = functools.partial(Normlayer, affine=True)
        self.main = nn.Sequential(
            # batch_size x in_channels x 64 x 64
            nn.Conv2d(in_channels, filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            bin(filters),
            nn.ReLU(True),
            nn.Conv2d(filters, filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            bin(filters)
            # batch_size x filters x 64 x 64
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != filters:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, filters, kernel_size=1, stride=stride, bias=False),
                bin(filters)
            )

    def forward(self, inputs):
        output = self.main(inputs)
        output += self.shortcut(inputs)
        return output

class D_AdaIN_ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, targets):
        super().__init__()
        mid_ch = min(in_ch, out_ch)
        self.conv1 = spectral_norm(nn.Conv2d(in_ch, mid_ch, 3, 1, 1))
        self.D_adain1 = D_AdaIN(mid_ch, targets)
        self.conv2 = spectral_norm(nn.Conv2d(mid_ch, out_ch, 3, 1, 1))
        self.D_adain2 = D_AdaIN(out_ch, targets)
        
        self.conv_s = spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False))
        self.D_adain_s = D_AdaIN(out_ch, targets)

    def forward(self, feature, target_mean, target_std, target):
        x_s = self.D_adain_s(self.conv_s(feature), target_mean, target_std, target)  # shortcut
        dx = self.conv1(feature)
        dx = self.conv2(F.relu(self.D_adain1(dx, target_mean, target_std, target)))
        dx = self.D_adain2(dx, target_mean, target_std, target)
        return F.relu(x_s + dx)

class D_AdaIN(nn.Module):
    def __init__(self, in_ch, targets):
        super().__init__()
        self.IN = nn.InstanceNorm2d(in_ch)
        self.mlp_mean = nn.ModuleDict()
        self.mlp_std = nn.ModuleDict()
        for dset in targets:
            self.mlp_mean[dset] = nn.Linear(64, 64)
            self.mlp_std[dset] = nn.Linear(64, 64)

    def forward(self, feature, target_mean, target_std, target):
        return self.mlp_std[target](target_std).unsqueeze(-1).unsqueeze(-1)*self.IN(feature) + self.mlp_mean[target](target_mean).unsqueeze(-1).unsqueeze(-1)