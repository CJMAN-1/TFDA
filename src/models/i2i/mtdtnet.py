import torch.nn as nn
import functools
from torch.nn.utils import spectral_norm
import torchvision
import torch.nn.functional as F
import torch
import logging

class Mtdtnet(nn.Module):
    def __init__(self,
                 encoder,
                 generator,
                 st_encoder,
                 discrminator,
                 domain_transfer,
                 label_embed,
                 datasets,  # 0: source 1~: targets
                 pretrained_mtdtnet=None):
        super(Mtdtnet, self).__init__()
        self.encoder = encoder
        self.generator = generator
        self.st_encoder = st_encoder
        self.discrminator = discrminator
        self.domain_transfer = domain_transfer
        self.label_embed = label_embed
        self.datasets = datasets
        self.converts = [ f'{datasets[0]}2{t}'for t in datasets[1:] ]
        print(self.datasets)
        print(self.converts)
        assert 0

        self.init_weights(pretrained_mtdtnet)
    
    def init_weights(self, pretrained=None):
        logger = logging.getLogger()
        if pretrained is not None:
            logger.info(f'load mtdtnet from: {pretrained}')
            self.load_state_dict(torch.load(pretrained))

    def _forward_train(self, imgs, labels):
        direct_recon, indirect_recon = dict(), dict()
        converted_imgs = dict()
        gamma, beta = dict(), dict()
        features = dict()
        D_outputs_fake = dict()
        
        pass
    
    def _forward_infer(self, imgs, labels):
        pass

    def forward(self, imgs, labels, mode=None):
        if mode == 'train' or mode is None:
            return self._forward_train(imgs, labels)
        elif mode == 'infer':
            return self._forward_infer(imgs, labels)

class Encoder(nn.Module):
    def __init__(self, channels=3):
        super(Encoder, self).__init__()
        bin = functools.partial(nn.GroupNorm, 4)
        # bin = functools.partial(Normlayer, affine=True)
        self.Encoder_Conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=4, stride=2, padding=1, bias=True),
            bin(32),
            nn.ReLU(True),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
        )

    def forward(self, inputs):
        # output = F.normalize(self.Encoder_Conv(inputs), dim=0) # batch_size x 512 x 10 x 6
        output = self.Encoder_Conv(inputs)
        return output


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.Decoder_Conv = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True)),
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
            nn.Conv2d(channels, 32, kernel_size=4, stride=2, padding=1, bias=True),
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
            # nn.GroupNorm(4, 64),
            nn.LeakyReLU(negative_slope=0.2),
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True)),  # 
            # nn.InstanceNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.Patch = nn.Sequential(
            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True)),  # 
            # nn.InstanceNorm2d(256),
            # nn.GroupNorm(4, 256),
            nn.LeakyReLU(negative_slope=0.2),
            spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=True)), # 
            # nn.InstanceNorm2d(512),
            # nn.GroupNorm(4, 512),
            nn.LeakyReLU(negative_slope=0.2),
            spectral_norm(nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1, bias=True)),
        )

        self.fc = nn.Sequential(
            spectral_norm(nn.Linear(67*60*128, 500)),
            nn.ReLU(),
            spectral_norm(nn.Linear(500, num_domains))
        )

    def forward(self, inputs):
        conv_output = self.Conv(inputs)
        patch_output = self.Patch(conv_output)
        fc_output = self.fc(conv_output.view(conv_output.size(0), -1))
        # fc_output = self.fc(conv_output).view(conv_output.size(0), -1)
        return (patch_output, fc_output)


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
        return self.embed(F.interpolate(seg.unsqueeze(1).float(), size=(270, 480), mode='nearest'))

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
