import torchvision
import torch.nn as nn

class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = torchvision.models.vgg19(pretrained=True).features
        self.to_relu_1_1 = nn.Sequential()
        self.to_relu_2_1 = nn.Sequential()
        self.to_relu_3_1 = nn.Sequential()
        self.to_relu_4_1 = nn.Sequential()
        self.to_relu_4_2 = nn.Sequential()

        for x in range(2):
            self.to_relu_1_1.add_module(str(x), features[x])
        for x in range(2,7):
            self.to_relu_2_1.add_module(str(x), features[x])
        for x in range(7,12):
            self.to_relu_3_1.add_module(str(x), features[x])
        for x in range(12,21):
            self.to_relu_4_1.add_module(str(x), features[x])
        for x in range(21,25):
            self.to_relu_4_2.add_module(str(x), features[x])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_1(x)
        h_relu_1_1 = h
        h = self.to_relu_2_1(h)
        h_relu_2_1 = h
        h = self.to_relu_3_1(h)
        h_relu_3_1 = h
        h = self.to_relu_4_1(h)
        h_relu_4_1 = h
        h = self.to_relu_4_2(h)
        h_relu_4_2 = h
        out = (h_relu_1_1, h_relu_2_1, h_relu_3_1, h_relu_4_1, h_relu_4_2)
        return out