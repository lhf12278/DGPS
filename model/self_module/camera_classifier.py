from torch.autograd import Function
from typing import Any, Optional, Tuple
import torch.nn as nn
import torch


class GradientReverseFunction(Function):
    """
    重写自定义的梯度计算方式
    """

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


class GRL_Layer(nn.Module):
    def __init__(self):
        super(GRL_Layer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0.0)


class domain_classfier(nn.Module):
    def __init__(self, in_dim, camids):
        super(domain_classfier, self).__init__()
        classifier = []
        classifier.append(nn.BatchNorm1d(in_dim))
        classifier.append(nn.ReLU(True))
        classifier.append(nn.Linear(in_dim, 100))
        classifier.append(nn.BatchNorm1d(100))
        classifier.append(nn.ReLU(True))
        classifier.append(nn.Linear(100, camids))
        self.classifier = nn.Sequential(*classifier)
        self.classifier.apply(weights_init_kaiming)
        self.grl = GRL_Layer()

    def forward(self, feature):
        grl_feature = self.grl(feature)
        domain_output = self.classifier(grl_feature)
        return domain_output