# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import torch
import torch.nn as nn

from utils.Utils import plot_feature_grid
torch.backends.cudnn.enabled = False
import torch.nn.functional as F
import math
# from networks.MINE import Mine, mutual_info_loss
from networks.encoder import build_cls
from networks.supermasks import  SuperMask
from networks.segbone import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5


class SAM(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAM, self).__init__()
        # self.max_pool = nn.AdaptiveMaxPool2d(output_size=1)
        # self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.sigmoid = nn.Sigmoid()
        # Spatial Attention
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1, padding=kernel_size//2,
                              bias=False)

    def forward(self, x):
        # Spatial Attention
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        mean_out = torch.mean(x, dim=1, keepdim=True)
        out = torch.cat((max_out, mean_out), dim=1)
        out = self.sigmoid(self.conv(out))
        out = out * x
        return out


class ACSAM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ACSAM, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 192, kernel_size=3, dilation=1, padding=1),
            nn.Conv2d(192, 192, kernel_size=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels, 192, kernel_size=3, dilation=1, padding=1),
            nn.Conv2d(192, 192, kernel_size=3, dilation=3, padding=3),
            nn.Conv2d(192, 192, kernel_size=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels, 192, kernel_size=3, dilation=1, padding=1),
            nn.Conv2d(192, 192, kernel_size=3, dilation=2, padding=2),
            nn.Conv2d(192, 192, kernel_size=3, dilation=3, padding=3),
            nn.Conv2d(192, 192, kernel_size=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels, 192, kernel_size=3, dilation=1, padding=1),
            nn.Conv2d(192, 192, kernel_size=3, dilation=3, padding=3),
            nn.Conv2d(192, 192, kernel_size=3, dilation=5, padding=5),
            nn.Conv2d(192, 192, kernel_size=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        self.sam = SAM(7)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + 768, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x)
        x4 = self.layer4(x)
        x5 = self.sam(x)
        out = torch.cat([x1, x2, x3, x4, x5], dim=1)
        out = self.conv(out)
        return out


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
    
class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, num_classes=2, num_domain=3, in_channels=[32, 64, 160, 256], embedding_dim=768, dropout_ratio=0.1):
        super(SegFormerHead, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            c1=embedding_dim*4,
            c2=embedding_dim,
            k=1,
        )

        self.linear_pred  = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.dropout  = nn.Dropout2d(dropout_ratio)
        
        self.linear_pred2  = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
    

        self.f1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.f2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.f3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.f4 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.f1.data.fill_(1.0)  # 1.0
        self.f2.data.fill_(1.0)
        self.f3.data.fill_(1.0)
        self.f4.data.fill_(1.0)

        self.mask = SuperMask(768)
        # build encoder for domain code
        self.encoder_d = build_cls(num_domain)
        
        self.acsam = ACSAM(embedding_dim, embedding_dim)
        
        self.acsam2 = ACSAM(embedding_dim, embedding_dim)
        self.fuse = ConvModule(
            c1=embedding_dim*2,
            c2=embedding_dim,
            k=1,
        )

    
    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        n, _, h, w = c4.shape
        
        c1 = self.f1 * c1
        
        c2 = self.f2 * c2
        c3 = self.f3 * c3
        c4 = self.f4 * c4
        

        ############## MLP decoder on C1-C4 ###########
        
        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])
        
       
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        

        ####################
        _c_ = _c.detach()
        re, ir, mask = self.mask.forward(_c_)
        
        domain_code = self.encoder_d(ir)
        re_p = self.linear_pred2(re)   ####_best
        
        idx = torch.randperm(ir.size(0))
        # idx = torch.tensor([3, 0, 2, 1])
        ir_shuffled = ir[idx]
        s_c = re + ir_shuffled

        x = self.linear_pred(self.dropout(self.acsam(_c)))
        x_ = self.linear_pred(self.dropout(self.acsam(s_c))) #best
     
        re_p = s_c

        return x_, x, domain_code, mask, re, ir, re_p

class SegFormer(nn.Module):
    def __init__(self, num_classes = 2, num_domain=3, phi = 'b0', pretrained = True):
        super(SegFormer, self).__init__()
        self.in_channels = {
            'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512], 'b2': [64, 128, 320, 512],
            'b3': [64, 128, 320, 512], 'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512],
        }[phi]
        self.backbone   = {
            'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2,
            'b3': mit_b3, 'b4': mit_b4, 'b5': mit_b5,
        }[phi](pretrained)
        self.embedding_dim   = {
            'b0': 256, 'b1': 256, 'b2': 768,
            'b3': 768, 'b4': 768, 'b5': 768,
        }[phi]
        self.decode_head = SegFormerHead(num_classes, num_domain, self.in_channels, self.embedding_dim)


    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)
        
        x = self.backbone.forward(inputs)
        
        x_, xr, domain_code, mask, re, ir, re_p = self.decode_head.forward(x)
      
        x_ = F.interpolate(x_, size=(H, W), mode='bilinear', align_corners=True)
        xr = F.interpolate(xr, size=(H, W), mode='bilinear', align_corners=True)
        re_p = F.interpolate(re_p, size=(H, W), mode='bilinear', align_corners=True)
        
        return  x_,xr, domain_code, mask, re, ir, re_p
