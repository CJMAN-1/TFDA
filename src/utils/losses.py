import torch
import torch.nn.functional as F
import torch.nn as nn
from src.models.commons.vgg19 import VGG19
from src.utils.matrix import *

### 모델에 상관없이 사용되는 loss function들
class Base_losses():
    def __init__(self):
        pass

    def CrossEntropy2d(self, predict, target, class_weight=None):
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))

        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != 255)
        target = target[target_mask]
        if not target.data.dim():
            return torch.zeros(1)
        predict = predict.permute(0,2,3,1).contiguous() # contiguous : premute를해도 접근하는 인덱스만 바뀌지 실제 메모리에서 위치는 안바뀌는데 contiguous는 실제 메모리위치를 인접하게 바꿔줌. view같은 함수를 쓸때 메모리가 연속하지않으면 오류가 난다고함.
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c).contiguous()

        loss = F.cross_entropy(predict, target, weight=class_weight, reduction='mean')

        return loss 
    
    def WeightedCrossEntropy2d(self, predict, target):
        pass


### Mtdt net에서만 사용되는 loss functions.
class Mtdt_losses(Base_losses):
    def __init__(self, datasets):
        super(Mtdt_losses, self).__init__()
        self.source = datasets[0]
        self.targets = datasets[1:]
        self.n_targets = len(self.targets)
        self.vgg19 = None 
        self.loss_fns = {}
        self.loss_fns['CE'] = nn.CrossEntropyLoss()
        self.loss_fns['L1'] = nn.L1Loss()
        

    def Gan_g(self, fake):
        patch_gen_loss, domain_gen_loss = 0, 0
        self.alpha_gen_patch = 1. / self.n_targets
        self.alpha_gen_domain = 1. / self.n_targets

        for convert in fake.keys():
            _, target = convert.split('2')
            #patch_fake, domain_fake = fake[convert]
            patch_fake = fake[convert]
            b = patch_fake.size(0)
            patch_gen_loss += -patch_fake.mean()
           
            # for ddp. if a module does not contribute for loss value, error raised    
            # if len(self.targets) == 1:
            #     self.alpha_gen_domain = 0
            #     domain_gen_loss += torch.sum(domain_fake)
            # elif target == self.targets[0]:
            #     domain_gen_loss += self.loss_fns['CE'](domain_fake, torch.zeros(b, device=patch_fake.device).long())
            # elif target == self.targets[1]:
            #     domain_gen_loss += self.loss_fns['CE'](domain_fake, torch.ones(b, device=patch_fake.device).long())
            # else:
            #     domain_gen_loss += self.loss_fns['CE'](domain_fake, 2 * torch.ones(b, device=patch_fake.device).long())
        # return self.alpha_gen_patch * patch_gen_loss + self.alpha_gen_domain * domain_gen_loss
        return self.alpha_gen_patch * patch_gen_loss



    def Direct_recon(self, input_imgs, recon_imgs):
        recon_loss = 0
        self.alpha_d_recon = 10

        for dset in input_imgs.keys():
            if dset == 'S':
                recon_loss += 0.1*self.loss_fns['L1'](input_imgs[dset], recon_imgs[dset])
            else:
                recon_loss += self.loss_fns['L1'](input_imgs[dset], recon_imgs[dset])
        return self.alpha_d_recon * recon_loss


    def Indirect_recon(self, input_img, recon_img):
        self.alpha_id_recon = 10
        return self.alpha_id_recon * self.loss_fns['L1'](input_img, recon_img)


    def Consis(self, source_img, cvt_imgs):
        if self.vgg19 == None:
            self.vgg19 = VGG19().cuda()

        loss = 0
        features = dict()
        features[self.source] = self.vgg19(source_img)

        for cvt in cvt_imgs.keys():
            features[cvt] = self.vgg19(cvt_imgs[cvt])
            loss += F.mse_loss(features[self.source][-1], features[cvt][-1])

        return loss


    def Style(self, target_imgs, cvt_imgs):
        if self.vgg19 == None:
            self.vgg19 = VGG19().cuda()

        loss = 0
        features = dict()
        for name, img in target_imgs.items():
            features[name] = self.vgg19(img)

        for cvt, img in cvt_imgs.items():
            _, target = cvt.split('2')
            features[cvt] = self.vgg19(img)
            gram_target = [gram_mat(feat) for feat in features[target]]
            gram_convert = [gram_mat(feat) for feat in features[cvt]]
            for i in range(len(gram_convert)):
                loss += F.mse_loss(gram_target[i], gram_convert[i])

        return loss

    def Gan_d(self, real, fake, targets):
        patch_dis_loss, domain_dis_loss = 0, 0
        self.alpha_dis_patch = 1. / self.n_targets
        self.alpha_dis_domain = 1. / self.n_targets
        
        for dset in real.keys():
            # patch_real, domain_real = real[dset]
            patch_real = real[dset]
            b = patch_real.size(0)
            patch_dis_loss += F.relu(1. - patch_real).mean()
           
            # if len(self.targets) == 1:
            #     self.alpha_dis_domain = 0
            #     domain_dis_loss += torch.sum(domain_real)
            # elif dset == targets[0]:
            #     domain_dis_loss += self.loss_fns['CE'](domain_real, torch.zeros(b, device=patch_real.device).long())
            # elif dset == targets[1]:
            #     domain_dis_loss += self.loss_fns['CE'](domain_real, torch.ones(b, device=patch_real.device).long())
            # else:
            #     domain_dis_loss += self.loss_fns['CE'](domain_real, 2 * torch.ones(b, device=patch_real.device).long())

        for convert in fake.keys():
            # patch_fake, domain_fake = fake[convert]
            patch_fake = fake[convert]

            patch_dis_loss += F.relu(1. + patch_fake).mean()
           
            # if len(self.targets) == 1:
            #     self.alpha_dis_domain = 0
            #     domain_dis_loss += torch.sum(domain_fake) 
            # elif dset == targets[0]:
            #     domain_dis_loss += self.loss_fns['CE'](domain_fake, torch.zeros(b, device=patch_fake.device).long())
            # elif dset == targets[1]:
            #     domain_dis_loss += self.loss_fns['CE'](domain_fake, torch.ones(b, device=patch_fake.device).long())
            # else:
            #     domain_dis_loss += self.loss_fns['CE'](domain_fake, 2 * torch.ones(b, device=patch_fake.device).long())

        # return self.alpha_dis_patch * patch_dis_loss + self.alpha_dis_domain * domain_dis_loss
        return self.alpha_dis_patch * patch_dis_loss