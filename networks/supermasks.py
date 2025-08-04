import itertools
import os
import sys
import torch
import torchvision
import torch.utils.data
from skimage.feature import canny
import numpy as np

from pprint import pprint
from itertools import combinations

from torch import nn, optim
from torch.nn import functional as F
from torch.distributions import Bernoulli, RelaxedBernoulli
from torchvision import datasets, models, transforms

from utils.Utils import compute_iou, plot_iou_distribution, save_edge_tensor_as_image, visualize_top_bottom_iou

SMOOTH = 1e-6

# class SuperMask(nn.Module):
#     def __init__(self, channels,init_setting="scalar", init_scalar=1):
#         super(SuperMask, self).__init__()
#         # self.domain_list = domain_list
#         # self.features = features
#         self.channels = channels
#         self.init_setting = init_setting
#         self.init_scalar = init_scalar
#         self.batch_size = 3
#         self.initialized = True
        
#         self.mask_logits = nn.Parameter(torch.randn(1, self.channels, 1) * self.init_scalar, requires_grad=True)
#         # self.mask_logits = nn.Parameter(self.mask_logit.repeat(self.batch_size, 1, 1).detach())  # 互不共享梯度
#         # self.mask_logits = nn.Parameter(torch.ones(self.batch_size, channels, 1) * init_scalar, requires_grad=True)  # [B, C, 1]
#         # self.mask_logits = nn.Parameter(torch.randn(3, channels, 1) * 5, requires_grad=True)

#     def forward(self, features):
#         # if not self.initialized:
#         #     self.initialize_mask_logits(features, target)  # **在 forward 里计算 IoU 初始化**
#         #     self.initialized = True
#         #     mask_logits = self.mask_logits
            
#         B, C, H, W = features.shape
#         mask_logits = self.mask_logits.expand(B, -1, -1)
#         # probs = torch.sigmoid(self.mask_logits).squeeze()
#         # probs = torch.sigmoid(self.mask_logits).squeeze(2) 
#         # probs = 0.5 * (torch.tanh(self.mask_logits) + 1).squeeze(2)# [B, C]
        
#         # probs = torch.clamp(mask_logits * 0.2 + 0.7, 0, 1)
#         # probs = torch.clamp(mask_logits * 0.2 + 0.5, 0, 1) 
#         probs = torch.sigmoid(mask_logits/0.1)
#         # probs = torch.clamp(mask_logits * 0.2 + 0.7, 0, 1)
#         # plot_iou_distribution(probs)
#         mask = probs
#         # hard_mask = (probs > 0.5).float()
        
        
#         # # eps = 1e-4
#         # # hard_mask = (probs > (0 + eps)).float()
#         # soft_mask = probs
#         # mask = (hard_mask - soft_mask).detach() + soft_mask

#         relevant_features = features * mask.view(B, self.channels, 1, 1)
#         irrelevant_features = features * (1 - mask.view(B, self.channels, 1, 1))
        
#         # iou_scores = compute_iou(target, relevant_features)
#         # visualize_top_bottom_iou(target, relevant_features, iou_scores)

#         return (relevant_features, irrelevant_features, mask)
    
#     def initialize_mask_logits(self, features, target):
#         B, C, H, W = features.shape
#         # mask = (mask > threshold).float()
#         mask_resized = F.interpolate(target, size=(64, 64), mode="bilinear", align_corners=False)
#         mask_np = mask_resized.squeeze(0).cpu().numpy()  # 转换为 numpy 处理

#         batch_size = mask_np.shape[0]
#         cup_edges_list = [canny(mask_np[i, 0], sigma=1) for i in range(batch_size)]
#         disc_edges_list = [canny(mask_np[i, 1], sigma=1) for i in range(batch_size)]

#         mask_edges = np.array([np.logical_or(cup_edges_list[i], disc_edges_list[i]).astype(np.float32) 
#                                 for i in range(batch_size)])
#         mask_edges_tensor = torch.tensor(mask_edges, dtype=torch.float32, device=target.device).unsqueeze(1)
        
#         features = (features > features.mean(dim=[2, 3], keepdim=True)).float()  

#         intersection = (mask_edges_tensor * features).sum(dim=[2, 3])  
#         union = mask_edges_tensor.sum(dim=[2, 3]) + features.sum(dim=[2, 3]) - intersection  
#         iou = intersection / (union + 1e-6)
#         # self.mask_logits.data = torch.where(iou > 0, torch.full_like(iou, 1.0), torch.full_like(iou, -1.0))
#         save_edge_tensor_as_image(mask_edges_tensor[0], "edge.png")
#         iou_result = mask_overlap(iou)
#         for k, v in iou_result.items():
#             print(f"{k}: {v:.4f}")
#         self.mask_logits.data = iou.unsqueeze(2)
        
#         # num_neg_ones = torch.sum(self.mask_logits.data == 0).item()
#         # print(f"Number of 0 values: {num_neg_ones}")
        
#         self.mask_logits.data = self.mask_logits.data.max(dim=0, keepdim=True)[0]
    
# class SuperMask(nn.Module):
#     def __init__(self, channels, height, width, init_setting="scalar", init_scalar=1):
#         super(SuperMask, self).__init__()
#         self.channels = channels
#         self.height = height
#         self.width = width
#         self.init_setting = init_setting
#         self.init_scalar = init_scalar

#         # 让 mask_logits 的形状变成 [1, C, H, W]，适配整个 feature map
#         self.mask_logits = nn.Parameter(torch.randn(1, self.channels, self.height, self.width) * self.init_scalar)

#     def forward(self, features):
#         B, C, H, W = features.shape

#         # 这里对 mask_logits 进行 sigmoid 激活，让它变成 (0,1) 之间
#         # probs = torch.sigmoid(self.mask_logits)  # [1, C, H, W]
#         probs = torch.clamp(self.mask_logits * 0.2 + 0.5, 0, 1)

#         # 让 mask 保持 batch 兼容性
#         hard_mask = (probs > 0.3).float()
#         soft_mask = probs
#         mask = (hard_mask - soft_mask).detach() + soft_mask  # [1, C, H, W]

#         # 让 mask 适配 batch 维度
#         mask = mask.expand(B, -1, -1, -1)  # 扩展到 [B, C, H, W]

#         # 计算相关和不相关特征
#         relevant_features = features * probs
#         irrelevant_features = features * (1 - probs)

#         return relevant_features, irrelevant_features, mask

    # def sparsity(self, mask):
    #     return torch.mean(mask, dim=1)

    # def sparsity_penalty(self):
    #     sparse_pen = 0
    #     for _, v in self.super_mask_logits.items():
    #         sparse_pen += torch.sum(nn.Sigmoid()(v))
    #     return sparse_pen

    # def overlap_penalty(self):
    #     overlap_pen = 0
    #     domain_pairs = list(combinations(self.domain_list, 2))
    #     for pair in domain_pairs:
    #         dom1, dom2 = pair
    #         mask1 = nn.Sigmoid()(self.super_mask_logits[dom1])
    #         mask2 = nn.Sigmoid()(self.super_mask_logits[dom2])
    #         intersection = torch.sum(mask1 * mask2)
    #         union = torch.sum(mask1 + mask2 - mask1 * mask2)
    #         iou = (intersection + SMOOTH) / (union + SMOOTH)
    #         overlap_pen += iou
    #     overlap_pen /= len(domain_pairs)
    #     return overlap_pen

    # def mask_overlap(self, layer_name=""):
    #     if layer_name != "":
    #         prefix = layer_name + " : "
    #     else:
    #         prefix = ""
    #     domain_pairs = combinations(self.domain_list, 2)
    #     iou_overlap_dict = {}
    #     for pair in domain_pairs:
    #         mask_0 = nn.Sigmoid()(self.super_mask_logits[pair[0]])
    #         mask_1 = nn.Sigmoid()(self.super_mask_logits[pair[1]])
    #         mask_0 = mask_0 > 0.5
    #         mask_1 = mask_1 > 0.5
    #         intersection = (mask_0 & mask_1).float().sum()
    #         union = (mask_0 | mask_1).float().sum()
    #         iou = (intersection + SMOOTH) / (union + SMOOTH)
    #         iou_overlap_dict[
    #             prefix + pair[0] + ", " + pair[1] + " IoU-Ov"
    #         ] = iou.data.item()
    #     iou_overlap_dict[prefix + "overall IoU-Ov"] = np.mean(
    #         [x for x in list(iou_overlap_dict.values())]
    #     )
    #     return iou_overlap_dict
def mask_overlap(masks):
    """
    计算 batch 内 domain-specific masks 的 IoU 重合度。
    
    参数:
    - masks: Tensor, shape (batch_size, channels)
    
    返回:
    - iou_overlap_dict: 字典，存储不同 domain 之间的 IoU
    """
    batch_size, channels = masks.shape  # 形状 (3, 768)

    # Sigmoid 激活后 binarize 掩码
    soft_mask = torch.sigmoid(masks)  # 归一化到 (0, 1)
    hard_mask = (soft_mask > 0.5).float()  # 二值化

    # 计算所有 domain 组合的 IoU
    domain_pairs = list(itertools.combinations(range(batch_size), 2))
    iou_overlap_dict = {}

    for d1, d2 in domain_pairs:
        mask_1 = hard_mask[d1]  # shape (768,)
        mask_2 = hard_mask[d2]  # shape (768,)

        intersection = (mask_1 * mask_2).sum().item()  # 交集
        union = ((mask_1 + mask_2) > 0).float().sum().item()  # 并集

        iou = intersection / (union + 1e-6)  # 防止除 0
        iou_overlap_dict[f"Domain {d1} vs Domain {d2} IoU"] = iou

    # 计算总体 IoU（所有 pair 的均值）
    overall_iou = sum(iou_overlap_dict.values()) / len(domain_pairs)
    iou_overlap_dict["Overall IoU"] = overall_iou

    return iou_overlap_dict

# class SuperMask(nn.Module):
#     def __init__(self, channels, init_setting="scalar", init_scalar=1):
#         super(SuperMask, self).__init__()
#         self.channels = channels
#         self.init_setting = init_setting
#         self.init_scalar = init_scalar
#         self.batch_size = 3
        
#         self.mask_logits = nn.Parameter(torch.randn(2, self.channels, 1, 1), requires_grad=True)

#     def forward(self, features, conv_mode=False): 
#         mask_logits = torch.softmax(self.mask_logits / 0.1 , dim=0)
#         # mask_logits = self.mask_logits
#         # plot_iou_distribution(mask_logits)
#         relevant_features = features * mask_logits[0].view(1, *mask_logits[0].shape)
#         irrelevant_features = features * mask_logits[1].view(1, *mask_logits[1].shape)

#         return (relevant_features, irrelevant_features, mask_logits)

class SuperMask(nn.Module):
    def __init__(self, channels):
        super(SuperMask, self).__init__()
        self.channels = channels

        # 两个独立的mask logits，shape = [channels, 1, 1]
        self.relevant_logits = nn.Parameter(torch.randn(channels, 1, 1), requires_grad=True)
        self.irrelevant_logits = nn.Parameter(torch.randn(channels, 1, 1), requires_grad=True)

    def forward(self, features):
        # 用 sigmoid 激活得到 [0,1] mask
        relevant_mask = torch.sigmoid(self.relevant_logits/0.1)  # [C,1,1]
        irrelevant_mask = torch.sigmoid(self.irrelevant_logits/0.1)
        # plot_iou_distribution(relevant_mask)


        relevant_features = features * relevant_mask.unsqueeze(0)  
        irrelevant_features = features * irrelevant_mask.unsqueeze(0)

        return relevant_features, irrelevant_features, (relevant_mask, irrelevant_mask)
    



class Projector(nn.Module):
    def __init__(self, output_size=128):
        super(Projector, self).__init__()
        self.conv = nn.Conv2d(24, 64, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(65536, output_size)

    def forward(self, x_in):
        x = self.conv(x_in)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.normalize(x, dim=1)
        return x
    