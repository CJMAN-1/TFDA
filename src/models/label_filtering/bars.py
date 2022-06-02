import torch.nn as nn
import torch
import torch.nn.functional as F

class BARS(nn.Module):
    def __init__(self, datasets, class_num):
        super().__init__()
        self.class_num = class_num
        self.centroid = dict()
        self.n_sample = dict()
        self.source = datasets[0]
        self.targets = datasets[1:]
        self.converts = [self.source+'2'+target for target in self.targets]

        for convert in self.converts:
            source, target = convert.split('2')
            self.centroid[target] = 0
            self.n_sample[target] = 0
            self.centroid[convert] = 0
            self.n_sample[convert] = 0

    def forward(self, feature_s2t, feature_target, label_s2t, label_target, convert):
        source, target = convert.split('2')
        with torch.no_grad():
            # B x cls x 2048 x H x W    
            feature_s2t_cls = feature_s2t.unsqueeze(1).expand(-1, self.centroid[convert].size(0), -1, -1, -1)
            feature_target_cls = feature_target.unsqueeze(1).expand(-1, self.centroid[target].size(0), -1, -1, -1)
            centroid_s2t = self.centroid[convert].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            centroid_target = self.centroid[target].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            centroid_s2t = centroid_s2t.expand_as(feature_s2t_cls)
            centroid_target = centroid_target.expand_as(feature_target_cls)

            # B x cls x H x W
            # distance_target_target = torch.norm(feature_target_cls-centroid_target, p=2, dim=2)
            
            # B x H x W
            mask_s2t = torch.argmin(torch.norm(feature_s2t_cls-centroid_target, p=2, dim=2), dim=1)
            mask_target= torch.argmin(torch.norm(feature_target_cls-centroid_s2t, p=2, dim=2), dim=1)
            mask_s2t = F.interpolate(mask_s2t.float().unsqueeze(1), size=label_s2t.size()[1:], mode='nearest').squeeze(1).long()
            mask_target = F.interpolate(mask_target.float().unsqueeze(1), size=label_target.size()[1:], mode='nearest').squeeze(1).long()

        return mask_s2t, mask_target

    def update(self, feature, seg, domain):
        self.n_sample[domain] += 1
        new_centroid = self.region_wise_pooling(feature, seg)  # cls x 2048
        if self.n_sample[domain] == 1:
            self.centroid[domain] = new_centroid
        else:
            ### 바꾼코드 ema
            alpha = 0.95
            self.centroid[domain] = alpha*self.centroid[domain] + (1-alpha)*new_centroid
            ### 기존코드
            # self.centroid[domain] += ((new_centroid - self.centroid[domain]) / self.n_sample[domain]) 

    def region_wise_pooling(self, codes, seg):
        segmap = F.one_hot(seg+1, num_classes=self.class_num+1).permute(0,3,1,2)
        segmap = F.interpolate(segmap.float(), size=codes.size()[2:], mode='nearest')

        b_size = codes.shape[0]
        # h_size = codes.shape[2]
        # w_size = codes.shape[3]
        f_size = codes.shape[1]

        s_size = segmap.shape[1]

        codes_vector = torch.zeros((b_size, s_size, f_size), dtype=codes.dtype, device=codes.device)

        for i in range(b_size):
            for j in range(s_size):
                component_mask_area = torch.sum(segmap.bool()[i, j])

                if component_mask_area > 0:
                    codes_component_feature = codes[i].masked_select(segmap.bool()[i, j]).reshape(f_size,  component_mask_area).mean(1)
                    codes_vector[i][j] = codes_component_feature

        return codes_vector.mean(dim=0)[1:]