import torch
import torch.nn as nn
import torch.nn.functional as F   

def kl_divergence(p, q):
  
    log_p = torch.log(p + 1e-8)
    log_q = torch.log(q + 1e-8)
    return (p * (log_p - log_q)).sum(dim=-1).mean()

def mutual_info_kl_loss(re, ir, temperature=0.1):

    re_flat = re.view(re.size(0), re.size(1), -1)  
    ir_flat = ir.view(ir.size(0), ir.size(1), -1)  

    sim = (re_flat @ ir_flat.transpose(1, 2)) / temperature   
    p_joint = F.softmax(sim, dim=-1)  

    
    re_prob = F.softmax(re_flat / temperature, dim=-1)
    ir_prob = F.softmax(ir_flat / temperature, dim=-1)
    p_marginal = re_prob @ ir_prob.transpose(1, 2)  

    loss = kl_divergence(p_joint, p_marginal)
    return loss
    
