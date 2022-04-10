import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from .config import cfg as pose_config
from .pose_hrnet import get_pose_net
from .pose_processor import HeatmapProcessor2


class ScoremapComputer(nn.Module):

    def __init__(self, norm_scale):
        super(ScoremapComputer, self).__init__()

        # init skeleton model
        self.keypoints_predictor = get_pose_net(pose_config, False)
        self.keypoints_predictor.load_state_dict(torch.load(pose_config.TEST.MODEL_FILE))
        # self.heatmap_processor = HeatmapProcessor(normalize_heatmap=True, group_mode='sum', gaussion_smooth=None)
        self.heatmap_processor = HeatmapProcessor2(normalize_heatmap=True, group_mode='sum', norm_scale=norm_scale)

    def forward(self, x):
        heatmap = self.keypoints_predictor(x)  # before normalization
        scoremap, keypoints_confidence, keypoints_location = self.heatmap_processor(heatmap)  # after normalization
        # print(keypoints_location.shape, keypoints_confidence.shape, scoremap.shape)
        # print(keypoints_location[0, :, :])
        return scoremap.detach(), keypoints_confidence.detach(), keypoints_location.detach()


def compute_local_features(feature_maps, score_maps, keypoints_confidence):
    '''
    the last one is global feature
    :param config:
    :param feature_maps:
    :param score_maps:
    :param keypoints_confidence:
    :return:
    '''
    fbs, fc, fh, fw = feature_maps.shape
    sbs, sc, sh, sw = score_maps.shape
    assert fbs == sbs and fh == sh and fw == sw
    # print("feature_map_size", feature_maps.shape, "score_map_size", score_maps.shape)

    # get feature_vector_list
    feature_vector_list = []
    for i in range(sc + 1):
        if i < sc:  # skeleton-based local feature vectors
            score_map_i = score_maps[:, i, :, :].unsqueeze(1).repeat([1, fc, 1, 1])
            feature_vector_i = torch.sum(score_map_i * feature_maps, [2, 3])
            feature_vector_list.append(feature_vector_i)
        else:  # global feature vectors
            feature_vector_i = (
                        F.adaptive_avg_pool2d(feature_maps, 1) + F.adaptive_max_pool2d(feature_maps, 1)).squeeze()
            feature_vector_list.append(feature_vector_i)
            keypoints_confidence = torch.cat([keypoints_confidence, torch.ones([fbs, 1]).cuda()], dim=1)

    # compute keypoints confidence
    keypoints_confidence[:, sc:] = F.normalize(
        keypoints_confidence[:, sc:], 1, 1) * 1.0  # global feature score_confidence
    keypoints_confidence[:, :sc] = F.normalize(keypoints_confidence[:, :sc], 1,
                                               1) * 1.0  # partial feature score_confidence

    return feature_vector_list, keypoints_confidence

def exchangelr(feat_list, l2r, r2l, confid):
    new_feature_list = [feat_list[0]]
    confid = confid.unsqueeze(-1)
    j = 1
    for i in range(1, 13, 2):


        new_right = (l2r * feat_list[i]) + ((torch.ones_like(l2r) - l2r) * feat_list[i + 1])
        new_lift = (r2l * feat_list[i + 1]) + ((torch.ones_like(r2l) - r2l) * feat_list[i])
        # new_right = (confid[:, j].expand(32, 2048) * feat_list[i]) + (confid[:, j + 1].expand(32, 2048) * feat_list[i + 1])
        # new_lift = (confid[:, j].expand(32, 2048) * feat_list[i]) + (confid[:, j + 1].expand(32, 2048) * feat_list[i + 1])
        new_feature_list.append(new_lift)
        new_feature_list.append(new_right)
        # except:
        #     print('?24?')
        #     wwwwww = feat_list[0].size()
        j += 2

    return new_feature_list

def cycle_lr(feat_list, lr, rl):
    new_feature_list = [feat_list[0]]
    j = 1
    for i in range(1, 13, 2):
        new_right = lr(feat_list[i])       ###l2r
        new_left = rl(feat_list[i + 1])    ###r2l
        new_feature_list.append(new_left)
        new_feature_list.append(new_right)
        j += 2
    return new_feature_list

def cycle_rl_a(feat_list, lr, rl):
    new_feature_list = [feat_list[0]]
    j = 1
    for i in range(1, 13, 2):
        new_right = lr(feat_list[i])   ###l'2 ro
        new_left = rl(feat_list[i + 1])
        new_feature_list.append(new_left)
        new_feature_list.append(new_right)
        j += 2
    return new_feature_list

def sum_of_map(feature_map, score):
    # for i, one_score in enumerate(score):
    #     if i == 0:
    #         score_sum = one_score
    #     else:
    #         score_sum += one_score
    for i in range(13):
        if i == 0:
            score_sum = score[:, i, :, :]
        else:
            score_sum += score[:, i, :, :]
    fmap_size = feature_map.shape
    a = score_sum.unsqueeze(1).repeat([1, fmap_size[1], 1, 1])
    feature_map_masked = a * feature_map
    return feature_map_masked



