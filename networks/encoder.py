import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class EncoderDC(nn.Module):
    def __init__(self, Num_D, BatchNorm):
        super(EncoderDC, self).__init__()
        # if backbone == 'drn':
        #     inplanes = 512
        # elif backbone == 'mobilenet':
        #     inplanes = 320
        # else:
        #     inplanes = 2048
        inplanes = 256
        inplanes = 256
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.bn = BatchNorm(inplanes)
        self.relu = nn.ReLU()
        self.cls = nn.Conv2d(inplanes, Num_D, 1)
        self._init_weight()

    def forward(self, x):
        x = self.pool(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.cls(x)

        return torch.squeeze(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            # elif isinstance(m, SynchronizedBatchNorm2d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_encoderDC(Num_D, BatchNorm):
    return EncoderDC(Num_D, BatchNorm)


# class cls(nn.Module):
#     def __init__(self, Num_D):
#         super(cls, self).__init__()
#         # if backbone == 'drn':
#         #     inplanes = 512
#         # elif backbone == 'mobilenet':
#         #     inplanes = 320
#         # else:
#         #     inplanes = 2048
#         inplanes = 768
#         # inplanes = 1024
#         self.pool = nn.AdaptiveMaxPool2d((1, 1))
#         self.bn = nn.BatchNorm2d(inplanes)
#         self.relu = nn.ReLU()
#         self.cls = nn.Conv2d(inplanes, Num_D, 1)
#         self._init_weight()

#         # self.linear_c4 = MLP(input_dim=512, embed_dim=768)
#         # self.linear_c3 = MLP(input_dim=320, embed_dim=768)
#         # self.linear_c2 = MLP(input_dim=128, embed_dim=768)
#         # self.linear_c1 = MLP(input_dim=64, embed_dim=768)

#     def forward(self, x):
        
#         # pooled_features = []
#         # for feature in x:
#         #     pooled_feature = self.pool(feature)  # 对每个尺度特征进行池化，输出大小变成 [batch_size, channels, 1, 1]
#         #     pooled_features.append(pooled_feature)

#         # # 将所有池化后的特征图在通道维度（dim=1）拼接
#         # x = torch.cat(pooled_features, dim=1)  # [batch_size, channels * 4, 1, 1]

#         # # x = self.pool(_c)
#         # x = self.bn(x)
#         # x1=x
#         # x = self.relu(x)
#         # x = self.cls(x)
#         x = self.pool(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         x = self.cls(x)

#         return torch.squeeze(x)
    
class cls(nn.Module):
    def __init__(self, Num_D):
        super(cls, self).__init__()
        # if backbone == 'drn':
        #     inplanes = 512
        # elif backbone == 'mobilenet':
        #     inplanes = 320
        # else:
        #     inplanes = 2048
        inplanes = 768
        # inplanes = 304
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.bn = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU()
        self.cls = nn.Conv2d(inplanes, Num_D, 1)
        self._init_weight()

    def forward(self, x):
        x = self.pool(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.cls(x)
        return torch.squeeze(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            # elif isinstance(m, SynchronizedBatchNorm2d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_encoderDC(Num_D, BatchNorm):
    return EncoderDC(Num_D, BatchNorm)

def build_cls(Num_D):
    return cls(Num_D)