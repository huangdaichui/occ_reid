import torch
import torch.nn as nn
import torch.nn.functional as nf
import torchvision
from torchvision import transforms
from .bnneck import BNClassifier, Local_align, common_feature_computer, common_feature_computer_map
from model_keypoints import compute_local_features, ScoremapComputer, exchangelr, sum_of_map, cycle_lr, cycle_rl_a
import numpy as np
from non_local import NONLocalBlock2D


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
            nn.init.constant_(m.bias, 1.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Res50BNNeck(nn.Module):

    def __init__(self, class_num, pretrained=True):
        super(Res50BNNeck, self).__init__()

        self.class_num = class_num
        # backbone and optimize its architecture
        resnet = torchvision.models.resnet50(pretrained=pretrained)
        resnet.layer4[0].conv2.stride = (1,1)
        resnet.layer4[0].downsample[0].stride = (1,1)

        # cnn backbone
        self.resnet_conv = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.maxpool, # no relu
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.device = torch.device('cuda')
        # classifier
        self.classifier = BNClassifier(2048, self.class_num)
        self.classifier_common = BNClassifier(2048, self.class_num)
        self.classifier_max = BNClassifier(2048, self.class_num)
        self.classifier_sum = BNClassifier(2048, self.class_num)
        self.lr = nn.Linear(2048, 2048, bias=False)
        self.rl = nn.Linear(2048, 2048, bias=False)
        self.local_classifier_list = []
        for _ in range(13):
            self.classifier_part = BNClassifier(2048, self.class_num).to(self.device)
            self.local_classifier_list.append(self.classifier_part)
        self.scoremap_computer = ScoremapComputer(norm_scale=1.0).to(self.device)
        # self.scoremap_computer = nn.DataParallel(self.scoremap_computer).to(self.device)
        self.scoremap_computer = self.scoremap_computer.eval()
        self.l2r = nn.Parameter(torch.randn(32, 2048))
        self.r2l = nn.Parameter(torch.randn(32, 2048))
        self.local_align = Local_align(2048, class_num=13)
        self.conv1x1 = nn.Conv2d(2048, 1, 1, 1, 0)
        self.BN = nn.BatchNorm2d(1024)
        self.BN.apply(weights_init_kaiming)
        self.IN = nn.InstanceNorm2d(1024, affine=True)
        self.IN.apply(weights_init_kaiming)
        self.non_local1 = NONLocalBlock2D(2048)

    def inbn(self, split):

        out1_1 = self.IN(split[0].contiguous())
        out1_2 = self.BN(split[0].contiguous())
        out2_1 = self.BN(split[1].contiguous())
        out2_2 = self.IN(split[1].contiguous())
        out1 = torch.cat((out1_1, out2_1), 1)
        out2 = torch.cat((out1_2, out2_2), 1)
        out = (out1 + out2) / 2.

        return out

    def forward(self, x):
        feature_maps = self.resnet_conv(x)
        score_maps, keypoints_confidence, _ = self.scoremap_computer(x)          ###### 返回的分别是，热图，置信度，坐标
        feature_vector_list, keypoints_confidence = compute_local_features(
            feature_maps, score_maps, keypoints_confidence)


        # toPIL = transforms.ToPILImage()
        # for index_heatmap in range(score_maps.size()[1]):
        #     pic = toPIL(nf.relu((score_maps[1, index_heatmap, :, :]-0.007)* 150))
        #     pic.save('random{}.jpg'.format(index_heatmap))
        # torchvision.utils.save_image(score_maps[1, 1, :, :]*200, 'randomkk.jpg')


        feature_vector_list.pop()
        # feature_maps_for_common = feature_maps
        feature_maps = sum_of_map(feature_maps, score_maps)  ### 综合淹没

        # print(len(feature_vector_list))
        # new_feature_vector_list = exchangelr(feature_vector_list, torch.sigmoid(self.l2r), torch.sigmoid(self.r2l), keypoints_confidence)
        # print(len(feature_vector_list), '1212', len(new_feature_vector_list))
        new_feature_vector_list = feature_vector_list
        # if not self.training:
        #     for i, local_features in enumerate(new_feature_vector_list):
        #         if i == 0:
        #             local_features_tensor
        #     print('testing')

        result_list = []
        local_align_list = []
        # print('feature_vector_list[0].shape = ', new_feature_vector_list[0].shape)
        for i, v in enumerate(self.local_classifier_list):
            bned_local_feature, result = v(new_feature_vector_list[i])
            _, local_align_result = self.local_align(new_feature_vector_list[i])
            if not self.training:
                if i == 0:
                    bned_local_feature_tensor = torch.unsqueeze(bned_local_feature, -1)
                else:
                    bned_local_feature_tensor = torch.cat((bned_local_feature_tensor, torch.unsqueeze(bned_local_feature, -1)), dim=2)
            result_list.append(result)
            local_align_list.append(local_align_result)

        ex_feature_list = cycle_lr(new_feature_vector_list, self.lr, self.rl)  #########第一次交换
        ex_cl_result_list = []
        ex_cl_local_align_list = []
        for i, v in enumerate(self.local_classifier_list):
            bned_local_feature, result = v(ex_feature_list[i])
            _, local_align_result = self.local_align(ex_feature_list[i])
            if not self.training:
                if i == 0:
                    bned_local_feature_tensor = torch.unsqueeze(bned_local_feature, -1)
                else:
                    bned_local_feature_tensor = torch.cat((bned_local_feature_tensor, torch.unsqueeze(bned_local_feature, -1)), dim=2)
            ex_cl_result_list.append(result)
            ex_cl_local_align_list.append(local_align_result)

        ex_a_feature_list = cycle_rl_a(new_feature_vector_list, self.lr, self.rl)  ######  再次交换
        ex_a_cl_result_list = []
        ex_a_cl_local_align_list = []
        for i, v in enumerate(self.local_classifier_list):
            bned_local_feature, result = v(ex_a_feature_list[i])
            _, local_align_result = self.local_align(ex_a_feature_list[i])
            if not self.training:
                if i == 0:
                    bned_local_feature_tensor = torch.unsqueeze(bned_local_feature, -1)
                else:
                    bned_local_feature_tensor = torch.cat((bned_local_feature_tensor, torch.unsqueeze(bned_local_feature, -1)), dim=2)
            ex_a_cl_result_list.append(result)
            ex_a_cl_local_align_list.append(local_align_result)


        features = self.gap(feature_maps)
        features_max = self.gmp(feature_maps)
        features_sum = features + features_max
        split = torch.split(features_sum, 1024, 1)
        features_sum = self.inbn(split).squeeze(dim=2).squeeze(dim=2)
        features = self.gap(feature_maps).squeeze(dim=2).squeeze(dim=2)
        features_max = self.gmp(feature_maps).squeeze(dim=2).squeeze(dim=2)

        bned_features, cls_score = self.classifier(features)
        bned_features_max, cls_score_max = self.classifier_max(features_max)
        bned_features_sum, cls_score_sum = self.classifier_sum(features_sum)




        common_features = common_feature_computer_map(feature_maps, self.conv1x1, self.gap, self.non_local1)



        _, cls_score_common = self.classifier_common(common_features)
        if self.training:
            return features, features_max, cls_score, result_list, local_align_list, cls_score_common, cls_score_max, cls_score_sum, ex_cl_result_list, ex_cl_local_align_list, ex_a_cl_result_list, ex_a_cl_local_align_list
        else:
            keypoints_confidence = keypoints_confidence[:, :13]
            return bned_features_sum, bned_local_feature_tensor, keypoints_confidence

