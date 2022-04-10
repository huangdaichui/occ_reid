import torch.nn as nn
import torch

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

#




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
        feature = self.bn(x)
        if not self.training:
            return feature, None
        cls_score = self.classifier(feature)

        return feature, cls_score

class Local_align(nn.Module):
    '''bn + fc'''

    def __init__(self, in_dim, class_num):
        super(Local_align, self).__init__()

        self.in_dim = in_dim
        self.class_num = class_num

        self.bn = nn.BatchNorm1d(self.in_dim)
        self.bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(self.in_dim, self.class_num, bias=False)

        self.bn.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        feature = self.bn(x)
        if not self.training:
            return feature, None
        cls_score = self.classifier(feature)

        return feature, cls_score

def common_feature_computer(person_tensor):

    for i in range(0, 32, 4):
        tensor1 = computer_one_common_feature(person_tensor[i], person_tensor[i + 1])
        tensor2 = computer_one_common_feature(person_tensor[i + 1], person_tensor[i + 2])
        tensor3 = computer_one_common_feature(person_tensor[i + 2], person_tensor[i + 3])
        tensor4 = computer_one_common_feature(person_tensor[i + 3], person_tensor[i])
        one_ID_person_tensor = torch.cat((tensor1, tensor2, tensor3, tensor4), 0)
        if i == 0:
            batch_ID_features = one_ID_person_tensor
        else:
            batch_ID_features = torch.cat((batch_ID_features, one_ID_person_tensor), 0)

    return batch_ID_features


def computer_one_common_feature(tensor1, tensor2):

    f = torch.matmul(tensor1.unsqueeze(-1), tensor2.unsqueeze(-1).t())
    N = f.size(-1)
    f_div_C = f / N
    f, _ = torch.max(f_div_C, 1)
    W_tensor1 = tensor1 * f
    z = W_tensor1 + tensor1
    z = z.unsqueeze(0)


    return z

# def common_feature_computer_map(common_feature_computer_map, conv, gap):
#     for i in range(0, 32, 4):
#         gap = nn.AdaptiveAvgPool2d(1)
#         tensor1 = computer_one_common_feature1(common_feature_computer_map[i], common_feature_computer_map[i + 1], conv)
#         tensor2 = computer_one_common_feature1(common_feature_computer_map[i + 1], common_feature_computer_map[i + 2], conv)
#         tensor3 = computer_one_common_feature1(common_feature_computer_map[i + 2], common_feature_computer_map[i + 3], conv)
#         tensor4 = computer_one_common_feature1(common_feature_computer_map[i + 3], common_feature_computer_map[i], conv)
#
#         one_ID_person_tensor = torch.cat((tensor1, tensor2, tensor3, tensor4), 0)
#         if i == 0:
#             batch_ID_features = one_ID_person_tensor
#         else:
#             batch_ID_features = torch.cat((batch_ID_features, one_ID_person_tensor), 0)
#
#     batch_ID_features = gap(batch_ID_features).squeeze(dim=2).squeeze(dim=2)
#
#     return batch_ID_features

def common_feature_computer_map(common_feature_computer_map, conv, gap, non_local1):
    for i in range(0, 32, 4):
        gap = nn.AdaptiveAvgPool2d(1)
        tensor1 = computer_one_common_feature1(common_feature_computer_map[i + 1], common_feature_computer_map[i], conv, non_local1)
        tensor2 = computer_one_common_feature1(common_feature_computer_map[i + 2], common_feature_computer_map[i + 1], conv, non_local1)
        tensor3 = computer_one_common_feature1(common_feature_computer_map[i + 3], common_feature_computer_map[i + 2], conv, non_local1)
        tensor4 = computer_one_common_feature1(common_feature_computer_map[i], common_feature_computer_map[i + 3], conv, non_local1)

        one_ID_person_tensor = torch.cat((tensor1, tensor2, tensor3, tensor4), 0)
        if i == 0:
            batch_ID_features = one_ID_person_tensor
        else:
            batch_ID_features = torch.cat((batch_ID_features, one_ID_person_tensor), 0)

    batch_ID_features = gap(batch_ID_features).squeeze(dim=2).squeeze(dim=2)

    return batch_ID_features

def computer_one_common_feature1(map1, map2, conv, nl):
    map1 = map1.unsqueeze(0)
    map2 = map2.unsqueeze(0)
    map1 = nl(map1)
    map2 = nl(map2)
    map1_ = conv(map1)
    map2_ = conv(map2)
    map1_ = map1_.view(1, 1, -1).permute(0, 2, 1)
    map2_ = map2_.view(1, 1, -1)
    f = torch.matmul(map1_, map2_)
    N = f.size(-1)
    f_div_C = f / N
    f_div_C1 = f_div_C.permute(0, 2, 1)

    f, max_index = torch.max(f_div_C, 2)
    f = f.view(1, 16, 8).unsqueeze(1)
    z = map2 * f


    f1, _ = torch.max(f_div_C1, 2)
    f1 = f.view(1, 16, 8).unsqueeze(1)
    z1 = map1 * f1
    z = z + z1

    return z