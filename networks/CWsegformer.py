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

class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
    
class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
 
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
 
        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
 
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
 
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x


class MLP_Lin(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = self.proj(x)
        return x
    
class PFE_module(nn.Module):
    def __init__(self, embedding_dim=768):
        super(PFE_module, self).__init__()
        self.mlp1 = MLP(input_dim=embedding_dim, embed_dim=embedding_dim)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.pool_mlp1 = MLP_Lin(input_dim=4096, embed_dim=4096 // 16)
        self.pool_relu = nn.LeakyReLU()
        self.pool_mlp2 = MLP_Lin(input_dim=4096 // 16, embed_dim=4096)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x = self.mlp1(inputs)
        x_att = self.avg_pool(x)
        x_att = self.pool_mlp1(x_att.transpose(1, 2))
        x_att = self.pool_relu(x_att)
        x_att = self.pool_mlp2(x_att).transpose(1, 2)
        x_att = self.sigmoid(x_att)
        x_weighted = x * x_att.expand_as(x)
        out = x_weighted.permute(0, 2, 1).reshape(inputs.shape[0], x.shape[2], inputs.shape[2], -1)
        return out

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
        # self.MI = Mine()
        # self.pool = nn.AdaptiveMaxPool2d((1, 1)) #test分析
        
        self.acsam = ACSAM(embedding_dim, embedding_dim)
        
        self.acsam2 = ACSAM(embedding_dim, embedding_dim)
        self.fuse = ConvModule(
            c1=embedding_dim*2,
            c2=embedding_dim,
            k=1,
        )
        ##################
        # self.CBAM = CBAMLayer(768)
        # self.edge = EdgePredictor(768, 64, 2)
        # self.d1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # self.d2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # self.d1.data.fill_(0.0)  # 1.0
        # self.d2.data.fill_(1.0)
        
        # self.channel_2D_transformer = Channel2DTransformer(768, 4)
        # self.spatial_attention = SpatialAttentionModule()

        # self.feature_Attention_c = PFE_module()
        # self.Att_c1 = MLP(input_dim=embedding_dim, embed_dim=embedding_dim)
        # self.Att_c2 = MLP(input_dim=embedding_dim, embed_dim=embedding_dim)
        # self.Att_c3 = MLP(input_dim=embedding_dim, embed_dim=embedding_dim)
        # self.Att_c4 = MLP(input_dim=embedding_dim, embed_dim=embedding_dim)
    
    def forward(self, inputs, target):
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
        
        #####################################################################################
        # pfe_c2 = self.feature_Attention_c(_c2)
        # pfe_c3 = self.feature_Attention_c(_c3)
        # pfe_c4 = self.feature_Attention_c(_c4)
        # pfe_c1 = self.spatial_attention(_c4)
        # re, ir, mask = self.mask.forward(_c1)
        
        # domain_code = self.encoder_d(ir)

        # idx = torch.randperm(ir.size(0))
        # ir_shuffled = ir[idx]
        # s_c1 = re + ir_shuffled
        # plot_feature_grid(_c1, "vis/", "_c1")
        # _c1 = _c1 + pfe_c2
        # plot_feature_grid(_c1, "vis/", "_c1_a")
        # # _c4 = self.ln4(_c4)
        # _c4 = self.Att_c4(_c4).permute(0, 2, 1).reshape(n, -1, _c4.shape[2], _c4.shape[3])

        # _c2 = _c2 + pfe_c3
        # # _c3 = self.ln3(_c3)
        # _c3 = self.Att_c3(_c3).permute(0, 2, 1).reshape(n, -1, _c3.shape[2], _c3.shape[3])

        # _c3 = _c3 + pfe_c4
        # # _c2 = self.ln2(_c2)
        # _c2 = self.Att_c2(_c2).permute(0, 2, 1).reshape(n, -1, _c2.shape[2], _c2.shape[3])

        # _c1 = _c1 * pfe_c1
        # # _c1 = self.ln1(_c1)
        # _c1 = self.Att_c1(_c1).permute(0, 2, 1).reshape(n, -1, _c1.shape[2], _c1.shape[3])
        # multi_level_features = [_c1, _c2, _c3, _c4]

        # attention_features = self.channel_2D_transformer(multi_level_features)
 
        ###################################################################################
        # _c = self.linear_fuse(torch.cat(attention_features, dim=1))
        # plot_feature_grid(_c2, "vis/", "c1")
        # plot_feature_grid(_c3, "vis/", "c3")
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        
        # s_c = self.linear_fuse(torch.cat([_c4, _c3, _c2, s_c1], dim=1))

        ####################
        _c_ = _c.detach()
        re, ir, mask = self.mask.forward(_c_)
        
        domain_code = self.encoder_d(ir)
        re_p = self.linear_pred2(re)   ####_best
        
        idx = torch.randperm(ir.size(0))
        # idx = torch.tensor([3, 0, 2, 1])
        ir_shuffled = ir[idx]
        s_c = re + ir_shuffled
        # s_c = re + ir
        # plot_feature_grid(s_c, "vis/", "s_c")
        # _c = self.CBAM(_c)
        # plot_feature_grid(re, "vis/", "re")
        # decoder
        
        # x = self.linear_pred(self.dropout(_c))
        # x_ = self.linear_pred(self.dropout(s_c))
        x = self.linear_pred(self.dropout(self.acsam(_c)))
        x_ = self.linear_pred(self.dropout(self.acsam(s_c))) #best
        # x_  = self.linear_pred(self.acsam(s_c))
        re_p = s_c
        # domain_code = (0,0,0)
        
        # edge = self.linear_pred(self.dropout(re))

        # return x
        # return x_, x, domain_code, mask, re, ir ###best
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
        # self.mask1 = SuperMask(64)
        # self.mask2 = SuperMask(128)
        # self.mask3 = SuperMask(320)
        # self.mask4 = SuperMask(512)
        # # build encoder for domain code
        # self.encoder_d = build_cls(num_domain)
        # self.pool = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, inputs, target):
        H, W = inputs.size(2), inputs.size(3)
        
        x = self.backbone.forward(inputs)
        # x1 = x
        # re_0, ir_0, mask0 = self.mask1(x[0])
        # re_1, ir_1, mask1 = self.mask2(x[1])
        # re_2, ir_2, mask2 = self.mask3(x[2])
        # re_3, ir_3, mask3 = self.mask4(x[3])

        # re = [re_0, re_1, re_2, re_3]
        # ir = [ir_0, ir_1, ir_2, ir_3]
        # masks = [mask0, mask1, mask2, mask3]
        
        # #特征
        # pooled_features = []
        # for feature in x:
        #     pooled_feature = self.pool(feature)  # 对每个尺度特征进行池化，输出大小变成 [batch_size, channels, 1, 1]
        #     pooled_features.append(pooled_feature)
            
        # o = torch.cat(pooled_features, dim=1) 
        
        # pooled_features = []
        # for feature in re:
        #     pooled_feature = self.pool(feature)  
        #     pooled_features.append(pooled_feature)
            
        # r = torch.cat(pooled_features, dim=1) 
        # pooled_features = []
        # for feature in ir:
        #     pooled_feature = self.pool(feature) 
        #     pooled_features.append(pooled_feature)
            
        # i = torch.cat(pooled_features, dim=1)
        
        # x1 = x  # 可视化编码多尺度特征
        # domain_code = self.encoder_d(ir)
        x_, xr, domain_code, mask, re, ir, re_p = self.decode_head.forward(x, target)
        # x = self.decode_head.forward(x, target)
        
        x_ = F.interpolate(x_, size=(H, W), mode='bilinear', align_corners=True)
        xr = F.interpolate(xr, size=(H, W), mode='bilinear', align_corners=True)
        re_p = F.interpolate(re_p, size=(H, W), mode='bilinear', align_corners=True)
        
        return  x_,xr, domain_code, mask, re, ir, re_p

        
# class SegFormer(nn.Module):
#     def __init__(self, num_classes = 2, num_domain=3, phi = 'b0', pretrained = True):
#         super(SegFormer, self).__init__()
#         self.in_channels = {
#             'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512], 'b2': [64, 128, 320, 512],
#             'b3': [64, 128, 320, 512], 'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512],
#         }[phi]
#         self.backbone   = {
#             'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2,
#             'b3': mit_b3, 'b4': mit_b4, 'b5': mit_b5,
#         }[phi](pretrained)
#         self.embedding_dim   = {
#             'b0': 256, 'b1': 256, 'b2': 768,
#             'b3': 768, 'b4': 768, 'b5': 768,
#         }[phi]
#         self.decode_head = SegFormerHead(num_classes, num_domain, self.in_channels, self.embedding_dim)
 

#     def forward(self, inputs, target):
#         H, W = inputs.size(2), inputs.size(3)
#         x = self.backbone.forward(inputs)
#         xr, domain_code, mask, re, ir = self.decode_head.forward(x, target)

#         xr = F.interpolate(xr, size=(H, W), mode='bilinear', align_corners=True)

#         return  xr, domain_code, mask, re, ir