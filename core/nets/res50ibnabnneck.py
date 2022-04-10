import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from .bnneck import BNClassifier, Local_align, common_feature_computer, common_feature_computer_map
from os.path import realpath, dirname, join
from model_keypoints import compute_local_features, ScoremapComputer, exchangelr, sum_of_map


__all__ = ['ResNet', 'resnet50_ibn_a', 'resnet101_ibn_a',
           'resnet152_ibn_a']

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class IBN(nn.Module):
    def __init__(self, planes):
        super(IBN, self).__init__()
        half1 = int(planes / 2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, ibn=False, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if ibn:
            self.bn1 = IBN(planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        scale = 64
        self.inplanes = scale
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, scale, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(scale)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, scale, layers[0])
        self.layer2 = self._make_layer(block, scale * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, scale * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, scale * 8, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(scale * 8 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        ibn = True
        if planes == 512:
            ibn = False
        layers.append(block(self.inplanes, planes, ibn, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, ibn))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet50_ibn_a(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        # print(join(realpath(dirname(__file__)), '/models/resnet50_ibn_a.pth.tar'))
        # model.load_state_dict(torch.load(join(realpath(dirname(__file__)), 'models/resnet50_ibn_a.pth.tar')))
        model.load_state_dict(torch.load(join(realpath(dirname(__file__)), 'models/r50_ibn_a.pth')))
        print('successfully load imagenet pre-trained resnet50-ibn model')
    return model


def resnet101_ibn_a(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152_ibn_a(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


class Res50IBNaBNNeck(nn.Module):

    def __init__(self, class_num, pretrained=True):
        super(Res50IBNaBNNeck, self).__init__()

        self.class_num = class_num
        # backbone and optimize its architecture
        resnet = resnet50_ibn_a(pretrained=pretrained)
        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)

        # cnn backbone
        self.resnet_conv = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.maxpool,  # no relu
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveAvgPool2d(1)
        self.device = torch.device('cuda')

        # classifier
        self.classifier = BNClassifier(2048, self.class_num)
        self.classifier_max = BNClassifier(2048, self.class_num)
        self.classifier_sum = BNClassifier(2048, self.class_num)
        self.classifier_common = BNClassifier(2048, self.class_num)
        self.scoremap_computer = ScoremapComputer(norm_scale=1.0).to(self.device)
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

    def forward(self, x):
        feature_maps = self.resnet_conv(x)

        score_maps, keypoints_confidence, _ = self.scoremap_computer(x)  ###### 返回的分别是，热图，置信度，坐标
        feature_vector_list, keypoints_confidence = compute_local_features(
            feature_maps, score_maps, keypoints_confidence)
        feature_vector_list.pop()

        feature_maps = sum_of_map(feature_maps, score_maps)  ### 综合淹没

        features = self.gap(feature_maps).squeeze(dim=2).squeeze(dim=2)
        features_max = self.gmp(feature_maps).squeeze(dim=2).squeeze(dim=2)
        features_sum = features + features_max
        bned_features, cls_score = self.classifier(features)
        bned_features_max, cls_score_max = self.classifier_max(features_max)
        bned_features_sum, cls_score_sum = self.classifier_sum(features_sum)
        common_features = common_feature_computer_map(feature_maps, self.conv1x1, self.gap)
        _, cls_score_common = self.classifier_common(common_features)



        new_feature_vector_list = exchangelr(feature_vector_list, torch.sigmoid(self.l2r), torch.sigmoid(self.r2l))
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
                    bned_local_feature_tensor = torch.cat(
                        (bned_local_feature_tensor, torch.unsqueeze(bned_local_feature, -1)), dim=2)
            result_list.append(result)
            local_align_list.append(local_align_result)

        if self.training:
            # return features, cls_score, result_list, local_align_list, cls_score_common
            return features, features_max, cls_score, result_list, local_align_list, cls_score_common, cls_score_max, cls_score_sum
        else:
            keypoints_confidence = keypoints_confidence[:, :13]
            return bned_features_sum, bned_local_feature_tensor, keypoints_confidence