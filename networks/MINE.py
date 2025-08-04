import torch
import torch.nn as nn
import torch.nn.functional as F   

def kl_divergence(p, q):
  
    log_p = torch.log(p + 1e-8)
    log_q = torch.log(q + 1e-8)
    return (p * (log_p - log_q)).sum(dim=-1).mean()

def mutual_info_kl_loss(re, ir, temperature=0.1):
    # 展平空间维度
    re_flat = re.view(re.size(0), re.size(1), -1)  # (B, C, N)
    ir_flat = ir.view(ir.size(0), ir.size(1), -1)  # (B, C, N)

    # 计算联合分布
    sim = (re_flat @ ir_flat.transpose(1, 2)) / temperature   # 加温度
    p_joint = F.softmax(sim, dim=-1)  # (B, C, C)

    # 计算边缘分布
    re_prob = F.softmax(re_flat / temperature, dim=-1)
    ir_prob = F.softmax(ir_flat / temperature, dim=-1)
    p_marginal = re_prob @ ir_prob.transpose(1, 2)  # (B, C, C)

    # KL 作为 Loss
    loss = kl_divergence(p_joint, p_marginal)
    return loss
    
