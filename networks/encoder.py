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


    
class cls(nn.Module):
    def __init__(self, Num_D):
        super(cls, self).__init__()
   
        inplanes = 768
    
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

def build_cls(Num_D):
    return cls(Num_D)