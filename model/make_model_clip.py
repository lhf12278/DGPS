import torch
import torch.nn as nn
import numpy as np
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .self_module.Learnable_module import CrossPositionSample
_tokenizer = _Tokenizer()
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .self_module.part_attention_transformer import LCG
from .self_module.camera_classifier import domain_classfier,GRL_Layer
from .backbones.bnneck import BNClassifier, LocalBnClassifiers
from utils.model_complexity import compute_model_complexity


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(build_transformer, self).__init__()
        self.model_name = cfg.MODEL.NAME
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.MODEL.SIE_COE

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.proj_classifier = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.proj_classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0] - 16) // cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1] - 16) // cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)

        self.image_encoder = clip_model.visual
        self.learn_parameters = CrossPositionSample(cfg, self.num_classes)
        self.camera_classifier = domain_classfier(self.in_planes, camera_num)

    def forward(self, x, target=None, cam_label=None, view_label=None, get_positives=None, pre_train=None, ad_train=None, attack=None):
        if pre_train:
            image_features_last, image_features, image_features_proj = self.image_encoder(x)   # x11, x12, xproj

            feat = image_features[:,0]
            ft_proj = image_features_proj[:,0]
            bn_feat_proj = self.bottleneck_proj(ft_proj)  # 最后一层做测试
            cls_score = self.proj_classifier(bn_feat_proj)
            return ft_proj, feat, cls_score   # 线性层就是将原型跟原型映射到相同的特征空间

        elif get_positives:
            cross_camera_sample = self.learn_parameters(target)
            fake_image_features_last, fake_image_features, fake_image_features_proj = self.image_encoder(cross_camera_sample)
            fake_ft_proj = fake_image_features_proj[:,0]
            bn_fake_feat_proj = self.bottleneck_proj(fake_ft_proj)  # 最后一层做测试
            fake_cls_score = self.proj_classifier(bn_fake_feat_proj)
            person_prototype = self.proj_classifier.weight.data

            return fake_cls_score, fake_ft_proj, person_prototype
        elif ad_train:
            if attack:
                ad_cross_camera_sample, cls_ctx = self.learn_parameters(target, domain_info=True)
                fake_image_features_last, fake_image_features, fake_image_features_proj = self.image_encoder(
                    ad_cross_camera_sample)
                fake_ft = fake_image_features[:, 0]
                ad_fake_ft = fake_ft
                fake_feat_bn = self.bottleneck(ad_fake_ft)
                ad_cls_score = self.classifier(fake_feat_bn)
                return fake_feat_bn, ad_cls_score, ad_cross_camera_sample, cls_ctx

            else:
                # fake
                ad_cross_camera_sample, _ = self.learn_parameters(target, domain_info=True)
                fake_image_features_last, fake_image_features, fake_image_features_proj = self.image_encoder(ad_cross_camera_sample)
                fake_ft = fake_image_features[:, 0]
                fake_ft_proj = fake_image_features_proj[:, 0]
                fake_feat_bn = self.bottleneck(fake_ft)  # 最后一层做测试
                fake_cls_score = self.classifier(fake_feat_bn)
                fake_ft_proj_bn = self.bottleneck_proj(fake_ft_proj)
                fake_proj_score = self.proj_classifier(fake_ft_proj_bn)
                # person prototype
                person_prototype = self.proj_classifier.weight.data   # todo 考虑是否行人原型参与训练

                image_features_last, image_features, image_features_proj = self.image_encoder(x)
                ft = image_features[:, 0]
                ft_proj = image_features_proj[:, 0]
                ft_proj_bn = self.bottleneck_proj(ft_proj)
                proj_score = self.proj_classifier(ft_proj_bn)

                feat_bn = self.bottleneck(ft)
                cls_score = self.classifier(feat_bn)
                camera_score = self.camera_classifier(ft)

                return [ft, ft_proj, feat_bn], [cls_score, proj_score],\
                    [fake_ft, fake_ft_proj, fake_feat_bn], [fake_cls_score, fake_proj_score],\
                    person_prototype, camera_score

        elif self.training is not True:  # 测试
            image_features_last, image_features, image_features_proj = self.image_encoder(x)  # B, N, C
            ft = image_features[:, 0]
            ft_proj = image_features_proj[:, 0]
            feat_bn = self.bottleneck(ft)  # 最后一层做测试
            ft_proj_bn = self.bottleneck_proj(ft_proj)

            return torch.cat([feat_bn, ft_proj_bn], dim=1) if self.neck_feat == 'after' else torch.cat([ft, ft_proj], dim=1)
    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        # self.load_state_dict(param_dict)
        for i in param_dict:
            if 'classifier.weight' in i:
                continue
            if 'proj_classifier.weight' in i:
                continue
            if 'learn_parameters.learnable_person_info' in i:
                continue
            if 'learn_parameters.Disturbance_Domain_Data' in i:
                continue
            if 'camera_classifier.classifier.5.weight' in i:
                continue
            if 'camera_classifier.classifier.5.bias' in i:
                continue
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])

        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


def make_clip_model(cfg, num_class, camera_num, view_num):
    model = build_transformer(num_class, camera_num, view_num, cfg)
    total_params, total_flops = compute_model_complexity(model, (1, 3, 256, 128))
    print("Number of parameter: %.2fM" % (total_params / 1e6))
    print("total_flops: %.2f" % (total_flops / 1e9))
    return model


from .clip import clip


def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model


