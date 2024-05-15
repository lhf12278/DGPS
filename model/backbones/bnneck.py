import torch
import torch.nn as nn

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


class BNClassifier(nn.Module):
    '''bn + fc'''

    def __init__(self, in_dim, class_num):
        super(BNClassifier, self).__init__()

        self.in_dim = in_dim
        self.class_num = class_num

        self.bn = nn.BatchNorm1d(self.in_dim)
        self.bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(self.in_dim, self.class_num, bias=False)
        self.bn.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        bned_feature = self.bn(x)
        cls_score = self.classifier(bned_feature)
        return bned_feature, cls_score


class LocalBnClassifiers(nn.Module):
    def __init__(self, in_dim, class_num, branch_num):
        super(LocalBnClassifiers, self).__init__()

        self.in_dim = in_dim
        self.class_num = class_num
        self.branch_num = branch_num

        for i in range(self.branch_num):
            setattr(self, 'classifier_{}'.format(i), BNClassifier(self.in_dim, self.class_num))

    def forward(self, local_vector_list):

        assert len(local_vector_list) == self.branch_num

        # bnneck for each sub_branch_feature
        bned_local_feature_vector_list, cls_local_score_list = [], []
        for i in range(self.branch_num):
            feature_vector_i = local_vector_list[i]
            classifier_i = getattr(self, 'classifier_{}'.format(i))
            bned_feature_vector_i, cls_score_i = classifier_i(feature_vector_i)

            bned_local_feature_vector_list.append(bned_feature_vector_i)
            cls_local_score_list.append(cls_score_i)

        return bned_local_feature_vector_list, cls_local_score_list

