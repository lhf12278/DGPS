import torch
import torch.nn as nn


class CrossPositionSample(nn.Module):
    def __init__(self, cfg, num_class):
        super().__init__()
        H = cfg.INPUT.SIZE_TRAIN[0]
        W = cfg.INPUT.SIZE_TRAIN[1]
        channel_num = 3
        cls_vectors = torch.empty(num_class, channel_num, H, W, dtype=torch.float32)
        nn.init.normal_(cls_vectors, std=0.02)   # learning
        self.learnable_person_info = nn.Parameter(cls_vectors)
        self.Disturbance_Domain_Data = nn.Parameter(cls_vectors)  # learning

        self.num_class = num_class

    def forward(self, label, domain_info=None):
        if domain_info is None:
            cls_ctx = self.learnable_person_info[label]
            cross_camera_sample = cls_ctx
            return cross_camera_sample
        else:
            cls_ctx = self.learnable_person_info[label]
            domain = self.Disturbance_Domain_Data[label]
            cross_camera_sample = cls_ctx + domain
            # cross_camera_sample = cls_ctx
            return cross_camera_sample, cls_ctx

