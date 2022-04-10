'''
given an or a list of image(s)
extrct its/their features
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base import Base
from .nets import Res50BNNeck, Res50IBNaBNNeck, osnet_ain_x1_0


def build_extractor(config, use_cuda):
    return Extractor(config.cnnbackbone, config.image_size, config.model_path, use_cuda)

class Extractor(Base):
    '''
    given *RGB* image(s) in format of a list, each element is a numpy of size *[h,w,c]*, range *[0,225]*
    return their feature(s)(list), each element is a numpy of size [feat_dim]
    '''

    def __init__(self, cnnbackbone, image_size, model_path, use_cuda):

        self.mode = 'extract'
        self.cnnbackbone = cnnbackbone
        self.image_size = image_size
        self.model_path = model_path
        self.use_cuda = use_cuda
        self.pid_num = 1
        # init model
        self._init_device(use_cuda)
        self._init_model()
        # resume model
        self.resume_from_model(self.model_path)
        self.set_eval()

    def _init_device(self, use_cuda):
        if use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def _init_model(self):
        pretrained = False if self.mode != 'train' else True
        if self.cnnbackbone == 'res50':
            self.model = Res50BNNeck(class_num=self.pid_num, pretrained=pretrained)
        elif self.cnnbackbone == 'res50ibna':
            self.model = Res50IBNaBNNeck(class_num=self.pid_num, pretrained=pretrained)
        elif self.cnnbackbone == 'osnetain':
            self.model = osnet_ain_x1_0(num_classes=self.pid_num, pretrained=pretrained, loss='softmax')
        else:
            assert 0, 'cnnbackbone error, expect res50, res50ibna, osnetain'
        self.model = self.model.to(self.device)

    def np2tensor(self, image):
        '''
        convert a numpy *hwc* image *(0,255)*  to a torch.tensor *chw* image *(0,1)*
        Args:
            image(numpy): [h,w,c], in format of RGB, range [0, 255]
        '''
        assert isinstance(image, np.ndarray), "input must be a numpy array!"
        image = image.astype(np.float) / 255.
        image = image.transpose([2,0,1])
        image = torch.from_numpy(image).float()
        return image

    def resize_images(self, images, image_size):
        '''resize a batch of images to image_size'''
        images = F.interpolate(images, image_size, mode='bilinear', align_corners=True)
        return images

    def normalize_images(self, images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        '''
        Args:
            images(torch.tensor): size [bs, c, h, w], range [0,1]
        Return:
            images(torch.tensor): size [bs, c, h, w],
        '''
        mean = torch.tensor(mean).view([1, 3, 1, 1]).repeat([images.size(0), 1, images.size(2), images.size(3)]).to(self.device)
        std = torch.tensor(std).view([1, 3, 1, 1]).repeat([images.size(0), 1, images.size(2), images.size(3)]).to(self.device)
        images = (images - mean) / std
        return images

    def extract_list(self, image_list):
        '''
        given *RGB* image(s) in format of a list, each element is a numpy of size *[h,w,c]*, range *[0,225]*
        return their feature(s)(list), each element is a numpy of size [feat_dim]
        Args:
            image_list(list): every element is a numpy of size *[h,w,c]* format *RGB* and range *[0,255]*
        Return:
            feature_list(list): every element is a numpy of size [feature_dim]
        '''
        images = [self.resize_images(self.np2tensor(image).unsqueeze(0), self.image_size) for image in image_list]
        images = torch.cat(images, dim=0)
        images = images.to(self.device)
        images = self.normalize_images(images)
        with torch.no_grad():
            features = self.model(images)
        features = features.data.cpu().numpy()
        feature_list = [feature for feature in features]
        return feature_list

    # def extract_image(self, image):
    #     '''
    #     given an image, return its feature
    #     Args:
    #         image(torch.tensor): [c,h,w]
    #     Return:
    #         feature: [feature_dim]
    #     '''
    #     image = image.to(self.device)
    #     images = image.unsqueeze(0)
    #     images = self.resize_images(images, self.image_size)
    #     images = self.normalize_images(images)
    #     features = self.model(images)
    #     feature = features.squeeze(0)
    #     return feature
    #
    # def extract_images(self, images):
    #     '''
    #     given more than one image, return their feature
    #     Args:
    #         image(torch.tensor): [bs, c,h,w]
    #     Return:
    #         feature: [bs, feature_dim]
    #     '''
    #     images = images.to(self.device)
    #     images = self.resize_images(images, self.image_size)
    #     images = self.normalize_images(images)
    #     features = self.model(images)
    #     return features
