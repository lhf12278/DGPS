import torch
import torch.nn.functional as F


def bn_self_super_loss(ft, fake_ft):

    distance_matrix = -torch.mm(F.normalize(ft,p=2,dim=1),
                                F.normalize(fake_ft,p=2,dim=1).t().detach())   # 真实样本能够预测出虚假样本，迫使模型提到行人的特征
    loss = torch.mean(distance_matrix)

    return loss




