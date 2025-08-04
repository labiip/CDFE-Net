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

def mask_overlap(masks):
    """
    计算 batch 内 domain-specific masks 的 IoU 重合度。
    
    参数:
    - masks: Tensor, shape (batch_size, channels)
    
    返回:
    - iou_overlap_dict: 字典，存储不同 domain 之间的 IoU
    """
    batch_size, channels = masks.shape  

    soft_mask = torch.sigmoid(masks)  
    hard_mask = (soft_mask > 0.5).float()  

    domain_pairs = list(itertools.combinations(range(batch_size), 2))
    iou_overlap_dict = {}

    for d1, d2 in domain_pairs:
        mask_1 = hard_mask[d1]  
        mask_2 = hard_mask[d2]  

        intersection = (mask_1 * mask_2).sum().item()  
        union = ((mask_1 + mask_2) > 0).float().sum().item()  

        iou = intersection / (union + 1e-6)  # 防止除 0
        iou_overlap_dict[f"Domain {d1} vs Domain {d2} IoU"] = iou

    overall_iou = sum(iou_overlap_dict.values()) / len(domain_pairs)
    iou_overlap_dict["Overall IoU"] = overall_iou

    return iou_overlap_dict


class SuperMask(nn.Module):
    def __init__(self, channels):
        super(SuperMask, self).__init__()
        self.channels = channels

        self.relevant_logits = nn.Parameter(torch.randn(channels, 1, 1), requires_grad=True)
        self.irrelevant_logits = nn.Parameter(torch.randn(channels, 1, 1), requires_grad=True)

    def forward(self, features):
       
        relevant_mask = torch.sigmoid(self.relevant_logits/0.1)  
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
    