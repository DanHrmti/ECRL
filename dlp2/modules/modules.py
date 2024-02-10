"""
Basic modules and layers.
"""

# imports
import numpy as np
import math
# torch
import torch
import torch.nn.functional as F
import torch.nn as nn
from dlp2.utils.util_func import spatial_transform, generate_correlation_maps
# from utils.util_func import spatial_transform, generate_correlation_maps


# ResBlock from: https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, padding='zeros'):
    """3x3 convolution with padding"""
    # return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
    #                  padding=dilation, groups=groups, bias=False, dilation=dilation)
    if padding == 'zeros':
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=dilation, groups=groups, bias=False, dilation=dilation)
    else:
        return nn.Sequential(nn.ReplicationPad2d(1),
                             nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=3, stride=stride,
                                       padding=0, groups=groups, bias=False, dilation=dilation))


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, padding='replicate', norm_type='gn'):
        super(BasicBlock, self).__init__()
        norm_layer = nn.BatchNorm2d if norm_type == 'bn' else nn.GroupNorm
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        n_groups = 4 if (planes % 4 == 0) else 5
        self.conv1 = conv3x3(inplanes, planes, stride, padding=padding)
        self.bn1 = norm_layer(planes) if norm_type == 'bn' else norm_layer(n_groups, planes, eps=1e-4)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, padding=padding)
        self.bn2 = norm_layer(planes) if norm_type == 'bn' else norm_layer(n_groups, planes, eps=1e-4)
        if downsample is not None:
            self.downsample = downsample
        elif stride > 1 or inplanes != planes:
            self.downsample = conv1x1(inplanes, planes, stride)
        else:
            self.downsample = None
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # print(f'out: {out.shape}, identity: {identity.shape}')
        out += identity
        out = self.relu(out)

        return out


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride=1, pad=0, pool=False, upsample=False, bias=False,
                 activation=True, batchnorm=True, relu_type='relu', pad_mode='replicate', use_resblock=False):
        super(ConvBlock, self).__init__()
        self.main = nn.Sequential()
        if use_resblock:
            # self.main.add_module(f'conv_{c_in}_to_{c_out}', ResidualBlock(c_in, c_out, padding=pad_mode))
            # n_groups = 4 if (c_out % 4 == 0) else 5
            # norm_layer = nn.GroupNorm(n_groups, c_out, eps=1e-4)
            self.main.add_module(f'conv_{c_in}_to_{c_out}',
                                 BasicBlock(c_in, c_out, stride=stride, padding=pad_mode))
        else:
            if pad_mode != 'zeros':
                self.main.add_module('replicate_pad', nn.ReplicationPad2d(pad))
                pad = 0
            self.main.add_module(f'conv_{c_in}_to_{c_out}', nn.Conv2d(c_in, c_out, kernel_size,
                                                                      stride=stride, padding=pad, bias=bias))
        if batchnorm and not use_resblock:
            # self.main.add_module(f'bathcnorm_{c_out}', nn.BatchNorm2d(c_out, eps=1e-4))
            n_groups = 4 if (c_out % 4 == 0) else 5
            self.main.add_module(f'grouupnorm_{c_out}', nn.GroupNorm(n_groups, c_out, eps=1e-4))
        if activation and not use_resblock:
            if relu_type == 'leaky':
                self.main.add_module(f'relu', nn.LeakyReLU(0.01))
            else:
                self.main.add_module(f'relu', nn.ReLU())
        if pool:
            self.main.add_module(f'max_pool2', nn.MaxPool2d(kernel_size=2, stride=2))
        # if upsample:
        #     self.main.add_module(f'upsample_nearest_2',
        #                          nn.Upsample(scale_factor=2, mode='nearest'))
        if upsample:
            self.main.add_module(f'upsample_bilinear_2',
                                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))

    def forward(self, x):
        y = self.main(x)
        return y


class KeyPointCNNOriginal(nn.Module):
    """
    CNN to extract heatmaps, inspired by KeyNet (Jakab et.al)
    """

    def __init__(self, cdim=3, channels=(32, 64, 128, 256), image_size=64, n_kp=8, pad_mode='replicate',
                 use_resblock=False, first_conv_kernel_size=7):
        super(KeyPointCNNOriginal, self).__init__()
        self.cdim = cdim
        self.image_size = image_size
        self.n_kp = n_kp
        cc = channels[0]
        ch = cc
        first_conv_pad = first_conv_kernel_size // 2
        self.main = nn.Sequential()
        self.main.add_module(f'in_block_1',
                             ConvBlock(cdim, cc, kernel_size=first_conv_kernel_size, stride=1,
                                       pad=first_conv_pad, pool=False, pad_mode=pad_mode,
                                       use_resblock=False, relu_type='relu'))
        self.main.add_module(f'in_block_2',
                             ConvBlock(cc, cc, kernel_size=3, stride=1, pad=1, pool=False, pad_mode=pad_mode,
                                       use_resblock=use_resblock, relu_type='relu'))

        sz = image_size
        for ch in channels[1:]:
            self.main.add_module('conv_in_{}_0'.format(sz), ConvBlock(cc, ch, kernel_size=3, stride=2, pad=1,
                                                                      pool=False, pad_mode=pad_mode,
                                                                      use_resblock=use_resblock, relu_type='relu'))
            self.main.add_module('conv_in_{}_1'.format(ch), ConvBlock(ch, ch, kernel_size=3, stride=1, pad=1,
                                                                      pool=False, pad_mode=pad_mode,
                                                                      use_resblock=use_resblock, relu_type='relu'))
            cc, sz = ch, sz // 2

        self.keymap = nn.Conv2d(channels[-1], n_kp, kernel_size=1)
        self.conv_output_size = self.calc_conv_output_size()
        num_fc_features = torch.zeros(self.conv_output_size).view(-1).shape[0]
        print("conv shape: ", self.conv_output_size)
        # print("num fc features: ", num_fc_features)
        # self.fc = nn.Linear(num_fc_features, self.fc_output)

    def calc_conv_output_size(self):
        dummy_input = torch.zeros(1, self.cdim, self.image_size, self.image_size)
        dummy_input = self.main(dummy_input)
        return dummy_input[0].shape

    def forward(self, x):
        y = self.main(x)
        # heatmap
        hm = self.keymap(y)
        return y, hm


class AlternativeSpatialSoftmaxKP(torch.nn.Module):
    """
    This module performs spatial-softmax (ssm) by performing marginalization over heatmaps.
    """

    def __init__(self, kp_range=(-1, 1)):
        super().__init__()
        self.kp_range = kp_range

    def forward(self, heatmap, probs=False, variance=False):
        batch_size, n_kp, height, width = heatmap.shape
        # p(x) = \int p(x,y)dy
        logits = heatmap.view(batch_size, n_kp, -1)  # [batch_size, n_kp, h * w]
        scores = torch.softmax(logits, dim=-1)  # [batch_size, n_kp, h * w]
        scores = scores.view(batch_size, n_kp, height, width)  # [batch_size, n_kp, h, w]
        y_axis = torch.linspace(self.kp_range[0], self.kp_range[1], height,
                                device=scores.device).type_as(scores).expand(1, 1, -1)  # [1, 1, features_dim_height]
        x_axis = torch.linspace(self.kp_range[0], self.kp_range[1], width,
                                device=scores.device).type_as(scores).expand(1, 1, -1)  # [1, 1, features_dim_width]

        # marginalize over x (width) and y (height)
        sm_h = scores.sum(dim=-1)  # [batch_size, n_kp, h]
        sm_w = scores.sum(dim=-2)  # [batch_size, n_kp, w]

        # # expected value: probability per coordinate * coordinate
        kp_h = torch.sum(sm_h * y_axis, dim=-1, keepdim=True)  # [batch_size, n_kp, 1]
        kp_h = kp_h.squeeze(-1)  # [batch_size, n_kp], y coordinate of each kp

        kp_w = torch.sum(sm_w * x_axis, dim=-1, keepdim=True)  # [batch_size, n_kp, 1]
        kp_w = kp_w.squeeze(-1)  # [batch_size, n_kp], x coordinate of each kp

        # stack keypoints
        kp = torch.stack([kp_h, kp_w], dim=-1)  # [batch_size, n_kp, 2], x, y coordinates of each kp

        if variance:
            # sigma^2 = E[x^2] - (E[x])^2
            y_sq = (scores * (y_axis.unsqueeze(-1) ** 2)).sum(dim=(-2, -1))  # [batch_size, n_kp]
            v_h = (y_sq - kp_h ** 2).clamp_min(1e-6)  # [batch_size, n_kp]
            x_sq = (scores * (x_axis.unsqueeze(-2) ** 2)).sum(dim=(-2, -1))  # [batch_size, n_kp]
            v_w = (x_sq - kp_w ** 2).clamp_min(1e-6)  # [batch_size, n_kp]

            # covariance: E[xy] - E[x]E[y]
            xy_sq = (scores * (y_axis.unsqueeze(-1) * x_axis.unsqueeze(-2))).sum(dim=(-2, -1))  # [batch_size, n_kp]
            cov = xy_sq - kp_h * kp_w

            # cov_2 = scores * (y_axis.unsqueeze(-1) - kp_h[:, :, None, None]) * (x_axis.unsqueeze(-2) - kp_w[:, :, None, None])
            # cov_2 = cov_2.sum(dim=(-2, -1))

            var = torch.stack([v_h, v_w, cov], dim=-1)
            # var = torch.stack([v_h, v_w], dim=-1)
            return kp, var
        if probs:
            return kp, sm_h, sm_w
        else:
            return kp


class CoordinateEncoding(nn.Module):
    """
    This module embeds keypoints via fourier features
    """

    def __init__(self, sigma=10.0, input_size=2, encoded_size=256, is_learned=False, is_random=True, b=None):
        super().__init__()
        self.sigma = sigma
        self.input_size = input_size
        self.encoded_size = encoded_size
        self.is_random = is_random
        self.is_learned = is_learned

        if self.is_random:
            b = self.sigma * torch.randn(encoded_size, input_size)
            if self.is_learned:
                self.b = nn.Parameter(b)
            else:
                self.register_buffer('b', b)
        else:
            if b is None:
                self.b = 1.0
            else:
                self.register_buffer('b', b)

    def forward(self, x):
        if self.is_random:
            vp = 2 * np.pi * x @ self.b.T
            return torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)
        else:
            vp = 2 * np.pi * x
            return torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)


class CNNDecoder(nn.Module):
    def __init__(self, cdim=3, channels=(64, 128, 256, 512, 512, 512), image_size=64, in_ch=16, n_kp=8,
                 pad_mode='zeros', use_resblock=False):
        super(CNNDecoder, self).__init__()
        self.cdim = cdim
        self.image_size = image_size
        cc = channels[-1]
        self.in_ch = in_ch
        self.n_kp = n_kp

        sz = 4

        self.main = nn.Sequential()
        self.main.add_module('depth_up',
                             ConvBlock(self.in_ch, cc, kernel_size=3, pad=1, upsample=True, pad_mode=pad_mode,
                                       use_resblock=use_resblock, batchnorm=False))
        for ch in reversed(channels[1:-1]):
            self.main.add_module('conv_to_{}'.format(sz * 2), ConvBlock(cc, ch, kernel_size=3, pad=1, upsample=True,
                                                                        pad_mode=pad_mode, use_resblock=use_resblock))
            cc, sz = ch, sz * 2

        # self.main.add_module('conv_to_{}'.format(sz * 2),
        #                      ConvBlock(cc, self.n_kp * (channels[0] // self.n_kp + 1), kernel_size=3, pad=1,
        #                                upsample=False,
        #                                pad_mode=pad_mode, use_resblock=use_resblock))
        # self.final_conv = ConvBlock(self.n_kp * (channels[0] // self.n_kp + 1), cdim, kernel_size=1, bias=True,
        #                             activation=False, batchnorm=False)

        self.main.add_module('conv_to_{}'.format(sz * 2),
                             ConvBlock(cc, channels[0], kernel_size=3, pad=1,
                                       upsample=False,
                                       pad_mode=pad_mode, use_resblock=False))
        self.final_conv = ConvBlock(channels[0], cdim, kernel_size=1, bias=True,
                                    activation=False, batchnorm=False, use_resblock=False)

    def forward(self, z, masks=None):
        y = self.main(z)
        if masks is not None:
            # masks: [bs, n_kp, feat_dim, feat_dim]
            bs, n_kp, fs, _ = masks.shape
            # y: [bs, n_kp * ch[0], feat_dim, feat_dim]
            y = y.view(bs, n_kp, -1, fs, fs)
            y = masks.unsqueeze(2) * y
            y = y.view(bs, -1, fs, fs)
        y = self.final_conv(y)
        y = torch.sigmoid(y)
        return y


class ImagePatcher(nn.Module):
    """
    Author: Tal Daniel
    This module take an image of size B x cdim x H x W and return a patchified tesnor
    B x cdim x num_patches x patch_size x patch_size. It also gives you the global location of the patch
    w.r.t the original image. We use this module to extract prior KP from patches, and we need to know their
    global coordinates for the Chamfer-KL.
    """

    def __init__(self, cdim=3, image_size=64, patch_size=16):
        super(ImagePatcher, self).__init__()
        self.cdim = cdim
        self.image_size = image_size
        self.patch_size = patch_size
        self.kh, self.kw = self.patch_size, self.patch_size  # kernel size
        self.dh, self.dw = self.patch_size, patch_size  # stride
        self.unfold_shape = self.get_unfold_shape()
        self.patch_location_idx = self.get_patch_location_idx()
        # print(f'unfold shape: {self.unfold_shape}')
        # print(f'patch locations: {self.patch_location_idx}')

    def get_patch_location_idx(self):
        h = np.arange(0, self.image_size)[::self.patch_size]
        w = np.arange(0, self.image_size)[::self.patch_size]
        ww, hh = np.meshgrid(h, w)
        hw = np.stack((hh, ww), axis=-1)
        hw = hw.reshape(-1, 2)
        return torch.from_numpy(hw).int()

    def get_patch_centers(self):
        mid = self.patch_size // 2
        patch_locations_idx = self.get_patch_location_idx()
        patch_locations_idx += mid
        return patch_locations_idx

    def get_unfold_shape(self):
        dummy_input = torch.zeros(1, self.cdim, self.image_size, self.image_size)
        patches = dummy_input.unfold(2, self.kh, self.dh).unfold(3, self.kw, self.dw)
        unfold_shape = patches.shape[1:]
        return unfold_shape

    def img_to_patches(self, x):
        patches = x.unfold(2, self.kh, self.dh).unfold(3, self.kw, self.dw)
        patches = patches.contiguous().view(patches.shape[0], patches.shape[1], -1, self.kh, self.kw)
        return patches

    def patches_to_img(self, x):
        patches_orig = x.view(x.shape[0], *self.unfold_shape)
        output_h = self.unfold_shape[1] * self.unfold_shape[2]
        output_w = self.unfold_shape[2] * self.unfold_shape[4]
        patches_orig = patches_orig.permute(0, 1, 2, 4, 3, 5).contiguous()
        patches_orig = patches_orig.view(-1, self.cdim, output_h, output_w)
        return patches_orig

    def forward(self, x, patches=True):
        # x [batch_size, 3, image_size, image_size] or [batch_size, 3, num_patches, image_size, image_size]
        if patches:
            return self.img_to_patches(x)
        else:
            return self.patches_to_img(x)


class VariationalKeyPointPatchEncoder(nn.Module):
    """
    This module encodes patches to KP via SSM. Additionally, we implement a variational version that encodes
    log-variance in addition to the mean, but we don't use it in practice, as constant prior std works better.
    We also experimented with extracting features directly from the patches for a prior for the visual features,
    but we didn't find it better than a constant prior (~N(0,1)). However, you can explore it by setting
    `learned_feature_dim`>0.
    """

    def __init__(self, cdim=3, channels=(16, 16, 32), image_size=64, n_kp=4, patch_size=16, kp_range=(0, 1),
                 pad_mode='replicate', sigma=0.1, dropout=0.0, learnable_logvar=False,
                 learned_feature_dim=0, use_resblock=False):
        super(VariationalKeyPointPatchEncoder, self).__init__()
        self.image_size = image_size
        self.dropout = dropout
        self.kp_range = kp_range
        self.use_resblock = use_resblock
        self.n_kp = n_kp  # kp per patch
        self.patcher = ImagePatcher(cdim=cdim, image_size=image_size, patch_size=patch_size)
        self.features_dim = int(patch_size // (2 ** (len(channels) - 1)))
        self.enc = KeyPointCNNOriginal(cdim=cdim, channels=channels, image_size=patch_size, n_kp=n_kp,
                                       pad_mode=pad_mode, use_resblock=self.use_resblock, first_conv_kernel_size=3)
        # self.ssm = SpatialLogSoftmaxKP(kp_range=kp_range) if use_logsoftmax else SpatialSoftmaxKP(kp_range=kp_range)
        self.ssm = AlternativeSpatialSoftmaxKP(kp_range=kp_range)
        self.sigma = sigma
        self.learnable_logvar = learnable_logvar
        self.learned_feature_dim = learned_feature_dim

        if self.learnable_logvar:
            self.to_logvar = nn.Sequential(nn.Linear(self.n_kp * (self.features_dim ** 2), 512),
                                           nn.ReLU(True),
                                           nn.Linear(512, 256),
                                           nn.ReLU(True),
                                           nn.Linear(256, self.n_kp * 2))  # logvar_x, logvar_y
        if self.learned_feature_dim > 0:
            self.to_features = nn.Sequential(nn.Linear(self.n_kp * (self.features_dim ** 2), 512),
                                             nn.ReLU(True),
                                             nn.Linear(512, 256),
                                             nn.ReLU(True),
                                             nn.Linear(256, self.n_kp * self.learned_feature_dim))  # logvar_x, logvar_y

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                # nn.init.normal_(m.weight, 0, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                # m.requires_grad_(False)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # m.requires_grad_(False)
                # use pytorch's default
                pass

    def img_to_patches(self, x):
        return self.patcher.img_to_patches(x)

    def patches_to_img(self, x):
        return self.patcher.patches_to_img(x)

    def get_global_kp(self, local_kp):
        # local_kp: [batch_size, num_patches, n_kp, 2]
        # returns the global coordinates of a KP within the original image.
        batch_size, num_patches, n_kp, _ = local_kp.shape
        global_coor = self.patcher.get_patch_location_idx().to(local_kp.device)  # [num_patches, 2]
        global_coor = global_coor[:, None, :].repeat(1, n_kp, 1)
        global_coor = (((local_kp - self.kp_range[0]) / (self.kp_range[1] - self.kp_range[0])) * (
                self.patcher.patch_size - 1) + global_coor) / (self.image_size - 1)
        global_coor = global_coor * (self.kp_range[1] - self.kp_range[0]) + self.kp_range[0]
        return global_coor

    def get_distance_from_patch_centers(self, kp, global_kp=False):
        # calculates the distance of a KP from the center of its parent patch. This is useful to understand (and filter)
        # if SSM detected something, otherwise, the KP will probably land in the center of the patch
        # (e.g., a solid-color patch will have the same activation in all pixels).
        if not global_kp:
            global_coor = self.get_global_kp(kp).view(kp.shape[0], -1, 2)
        else:
            global_coor = kp
        centers = 0.5 * (self.kp_range[1] + self.kp_range[0]) * torch.ones_like(kp).to(kp.device)
        global_centers = self.get_global_kp(centers.view(kp.shape[0], -1, self.n_kp, 2)).view(kp.shape[0], -1, 2)
        return ((global_coor - global_centers) ** 2).sum(-1)

    def encode(self, x, global_kp=False):
        # x: [batch_size, cdim, image_size, image_size]
        # global_kp: set True to get the global coordinates within the image (instead of local KP inside the patch)
        batch_size, cdim, image_size, image_size = x.shape
        x_patches = self.img_to_patches(x)  # [batch_size, cdim, num_patches, patch_size, patch_size]
        x_patches = x_patches.permute(0, 2, 1, 3, 4)  # [batch_size, num_patches, cdim, patch_size, patch_size]
        x_patches = x_patches.contiguous().view(-1, cdim, self.patcher.patch_size, self.patcher.patch_size)
        _, z = self.enc(x_patches)  # [batch_size*num_patches, n_kp, features_dim, features_dim]
        mu_kp, var_kp = self.ssm(z, probs=False, variance=True)  # [batch_size * num_patches, n_kp, 2]
        mu_kp = mu_kp.view(batch_size, -1, self.n_kp, 2)  # [batch_size, num_patches, n_kp, 2]
        var_kp = var_kp.view(batch_size, -1, self.n_kp, 2)  # [batch_size, num_patches, n_kp, 2]
        if global_kp:
            mu_kp = self.get_global_kp(mu_kp)
        if self.learned_feature_dim > 0:
            mu_features = self.to_features(z.view(z.shape[0], -1))
            mu_features = mu_features.view(batch_size, -1, self.n_kp, self.learned_feature_dim)
            # [batch_size, num_patches, n_kp, learned_feature_dim]
        if self.learnable_logvar:
            logvar_kp = self.to_logvar(z.view(z.shape[0], -1))
            logvar_kp = logvar_kp.view(batch_size, -1, self.n_kp, 2)  # [batch_size, num_patches, n_kp, 2]
            if self.learned_feature_dim > 0:
                return mu_kp, logvar_kp, mu_features
            else:
                return mu_kp, logvar_kp
        elif self.learned_feature_dim > 0:
            return mu_kp, mu_features
        else:
            return mu_kp, var_kp

    def forward(self, x, global_kp=False):
        return self.encode(x, global_kp)


class SelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, n_embed, n_head, attn_pdrop=0.0, resid_pdrop=0.0, scaled_attn=True):
        super().__init__()
        assert n_embed % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embed, n_embed)
        self.query = nn.Linear(n_embed, n_embed)
        self.value = nn.Linear(n_embed, n_embed)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embed, n_embed)
        self.n_head = n_head

        self.scaled_attn = scaled_attn

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        scale = math.sqrt(k.size(-1)) if self.scaled_attn else 1.0
        att = (q @ k.transpose(-2, -1)) * (1.0 / scale)
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, n_embed, resid_pdrop=0.0, hidden_dim_multiplier=1, activation='gelu'):
        super().__init__()
        self.fc_1 = nn.Linear(n_embed, hidden_dim_multiplier * n_embed)
        if activation == 'gelu':
            self.act = nn.GELU()
        else:
            self.act = nn.ReLU(True)
        self.proj = nn.Linear(hidden_dim_multiplier * n_embed, n_embed)
        self.dropout = nn.Dropout(resid_pdrop)

    def forward(self, x):
        x = self.dropout(self.proj(self.act(self.fc_1(x))))
        return x


class AttentionBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embed, n_head, attn_pdrop=0.0, resid_pdrop=0.0, hidden_dim_multiplier=2, activation='relu',
                 layer_norm=False, residual=True, scaled_attn=True):
        super().__init__()
        norm = nn.LayerNorm if layer_norm else nn.Identity
        self.ln1 = norm(n_embed)
        self.ln2 = norm(n_embed)
        self.attn = SelfAttention(n_embed, n_head, attn_pdrop, resid_pdrop, scaled_attn=scaled_attn)
        self.mlp = MLP(n_embed, resid_pdrop, hidden_dim_multiplier, activation=activation)
        self.residual = residual

    def forward(self, x):
        attn_out = self.attn(self.ln1(x))
        if self.residual:
            x = x + attn_out
        else:
            x = attn_out
        mlp_out = self.mlp(self.ln2(x))
        if self.residual:
            x = x + mlp_out
        else:
            x = mlp_out
        return x


class ParticleAttributeEncoder(nn.Module):
    """
    Glimpse-encoder: encodes patches visual features in a variational fashion (mu, log-variance).
    Useful for object-based scenes.
    """

    def __init__(self, anchor_size, image_size, cnn_channels=(16, 16, 32), margin=0, ch=3, max_offset=1.0,
                 kp_activation='tanh', use_resblock=False, use_correlation_heatmaps=False, enable_attn=False,
                 hidden_dims=(256, 128), attn_dropout=0.1):
        super().__init__()
        # use_correlation_heatmaps: use correlation heatmaps as input to model particle properties (e.g., xy offset)
        self.anchor_size = anchor_size
        self.channels = cnn_channels
        self.image_size = image_size
        self.patch_size = np.round(anchor_size * (image_size - 1)).astype(int)
        self.margin = margin
        self.crop_size = self.patch_size + 2 * margin
        self.ch = ch
        self.enable_attn = enable_attn
        self.use_resblock = use_resblock
        self.use_correlation_heatmaps = use_correlation_heatmaps
        self.kp_activation = kp_activation
        self.max_offset = max_offset  # max offset of x-y, [-max_offset, +max_offset]
        self.hidden_dims = hidden_dims
        hidden_dim_1 = hidden_dims[0]
        hidden_dim_2 = hidden_dims[1]

        in_ch = (ch + 1) if self.use_correlation_heatmaps else ch
        self.cnn = KeyPointCNNOriginal(cdim=in_ch, channels=cnn_channels, image_size=self.crop_size, n_kp=32,
                                       pad_mode='replicate', use_resblock=self.use_resblock,
                                       first_conv_kernel_size=3)

        feature_map_size = (self.crop_size // 2 ** (len(cnn_channels) - 1)) ** 2
        # fc_in_dim = 32 * ((self.crop_size // 4) ** 2) if cnn else self.ch * (self.crop_size ** 2)
        fc_in_dim = 32 * feature_map_size
        # self.backbone = nn.Sequential(nn.Linear(fc_in_dim, hidden_dim_1),
        #                               nn.ReLU(True),
        #                               nn.Linear(hidden_dim_1, hidden_dim_2),
        #                               nn.ReLU(True))
        self.projection = nn.Linear(fc_in_dim, hidden_dim_2)
        # attention
        if enable_attn:
            # self.projection = nn.Linear(hidden_dim_2, hidden_dim_2)
            # self.projection = nn.Linear(fc_in_dim, hidden_dim_2)
            self.pos_enc = CoordinateEncoding(encoded_size=hidden_dim_2 // 2, is_random=True, is_learned=False)
            self.pos_emb = nn.Sequential(nn.Linear(hidden_dim_2 + 2, 4 * hidden_dim_2),
                                         nn.GELU(),
                                         nn.Linear(4 * hidden_dim_2, hidden_dim_2)
                                         )
            self.self_attention = AttentionBlock(n_embed=hidden_dim_2, n_head=1, resid_pdrop=attn_dropout,
                                                 attn_pdrop=attn_dropout,
                                                 layer_norm=True)
        else:
            # self.projection = nn.Identity()
            self.pos_enc = nn.Identity()
            self.pos_emb = nn.Identity()
            self.self_attention = nn.Identity()

        # hidden_dim_3 = hidden_dim_2 * 2 if self.enable_attn else hidden_dim_2
        self.backbone = nn.Sequential(nn.Linear(hidden_dim_2, hidden_dim_1),
                                      nn.ReLU(True),
                                      nn.Linear(hidden_dim_1, hidden_dim_2),
                                      nn.ReLU(True))

        self.x_head = nn.Linear(hidden_dim_2, 2)  # mu_x, logvar_x
        self.y_head = nn.Linear(hidden_dim_2, 2)  # mu_y, logvar_y
        # self.offset_xy_head = nn.Linear(hidden_dim_2, self.n_kp_enc * 4)  # mu_off_x, logvar_off_x, same for y
        self.scale_xy_head = nn.Linear(hidden_dim_2, 4)  # mu_sx, logvar_sx, mu_sy, logvar_sy
        self.obj_on_head = nn.Linear(hidden_dim_2, 2)  # [log_obj_on_a, log_obj_on_b]
        self.depth_head = nn.Linear(hidden_dim_2, 2)  # mu_depth, logvar_depth

    def forward(self, x, kp, z_scale=None, previous_objects=None):
        # x: [bs, ch, image_size, image_size]
        # kp: [bs, n_kp, 2] in [-1, 1]
        # exclusive_objects: create cumulative masks to avoid overlapping objects, THIS WAS NOT USED IN THE PAPER
        #                    this will (mostly) enforce one particle pre object and
        #                    won't allow several layers per object
        # previous_objects: [bs * n_kp, ch, patch_size, patch_size]
        # previous_objects: used to create correlation maps with objects from the previous timesteps
        batch_size, _, _, img_size = x.shape
        _, n_kp, _ = kp.shape
        if self.use_correlation_heatmaps and self.cnn is not None:
            cropped_objects = generate_correlation_maps(x, kp, self.patch_size, previous_objects=previous_objects,
                                                        z_scale=z_scale)
            # [batch_size * n_kp, ch + 1, patch_size, patch_size]
        else:
            x_repeated = x.unsqueeze(1).repeat(1, n_kp, 1, 1, 1)  # [batch_size, n_kp, ch, image_size, image_size]
            x_repeated = x_repeated.view(-1, *x.shape[1:])  # [batch_size * n_kp, ch, image_size, image_size]
            if z_scale is None:
                z_scale = (self.patch_size / img_size) * torch.ones_like(kp)
            else:
                # assume unnormalized z_scale
                z_scale = torch.sigmoid(z_scale)
            z_pos = kp.reshape(-1, kp.shape[-1])
            z_scale = z_scale.view(-1, z_scale.shape[-1])
            out_dims = (batch_size * n_kp, x.shape[1], self.patch_size, self.patch_size)
            cropped_objects = spatial_transform(x_repeated, z_pos, z_scale, out_dims, inverse=False)
            # [batch_size * n_kp, ch, patch_size, patch_size]

        # encode objects - fc
        _, cropped_objects_cnn = self.cnn(cropped_objects)
        cropped_objects_flat = cropped_objects_cnn.reshape(batch_size, n_kp, -1)  # flatten
        # backbone features
        # backbone_features = self.backbone(cropped_objects_flat)
        backbone_features = cropped_objects_flat
        # projection
        backbone_features = self.projection(backbone_features)

        # attention
        if self.enable_attn:
            # projection
            # backbone_features = self.projection(backbone_features)
            # positional encoding
            kp_pos_enc = self.pos_emb(torch.cat([self.pos_enc(kp.detach()), kp.detach()], dim=-1))
            backbone_features_in = backbone_features + kp_pos_enc  # [batch_size, n_kp, hidden_dim_2]
            # attention + residual
            attn_out = self.self_attention(backbone_features_in)
            # backbone_features = backbone_features + attn_out
            # backbone_features = torch.cat([backbone_features, attn_out], dim=-1)
            backbone_features = attn_out

        # backbone features
        backbone_features = self.backbone(backbone_features)
        # projection to output
        stats_x = self.x_head(backbone_features)
        stats_x = stats_x.view(batch_size, n_kp, 2)
        mu_x, logvar_x = stats_x.chunk(chunks=2, dim=-1)

        stats_y = self.y_head(backbone_features)
        stats_y = stats_y.view(batch_size, n_kp, 2)
        mu_y, logvar_y = stats_y.chunk(chunks=2, dim=-1)

        mu = torch.cat([mu_x, mu_y], dim=-1)
        logvar = torch.cat([logvar_x, logvar_y], dim=-1)

        scale_xy = self.scale_xy_head(backbone_features)
        scale_xy = scale_xy.view(batch_size, n_kp, -1)
        mu_scale, logvar_scale = torch.chunk(scale_xy, chunks=2, dim=-1)

        if self.kp_activation == "tanh":
            mu = self.max_offset * torch.tanh(mu)
        elif self.kp_activation == "sigmoid":
            mu = torch.sigmoid(mu)

        obj_on = self.obj_on_head(backbone_features)
        obj_on = obj_on.view(batch_size, n_kp, 2)
        lobj_on_a, lobj_on_b = torch.chunk(obj_on, chunks=2, dim=-1)  # log alpha, beta of Beta dist
        lobj_on_a = lobj_on_a.squeeze(-1)
        lobj_on_b = lobj_on_b.squeeze(-1)

        depth = self.depth_head(backbone_features)
        depth = depth.view(batch_size, n_kp, 2)
        mu_depth, logvar_depth = torch.chunk(depth, 2, dim=-1)
        spatial_out = {'mu': mu, 'logvar': logvar, 'mu_scale': mu_scale, 'logvar_scale': logvar_scale,
                       'lobj_on_a': lobj_on_a, 'lobj_on_b': lobj_on_b, 'obj_on': obj_on,
                       'mu_depth': mu_depth, 'logvar_depth': logvar_depth, }
        return spatial_out


class ParticleFeaturesEncoder(nn.Module):
    """
    Glimpse-encoder: encodes patches visual features in a variational fashion (mu, log-variance).
    Useful for object-based scenes.
    """

    def __init__(self, anchor_size, features_dim, image_size, cnn_channels=(16, 16, 32), margin=0, ch=3,
                 use_resblock=False, enable_attn=False, hidden_dims=(256, 128), attn_dropout=0.1):
        super().__init__()
        # use_correlation_heatmaps: use correlation heatmaps as input to model particle properties (e.g., xy offset)
        self.anchor_size = anchor_size
        self.channels = cnn_channels
        self.image_size = image_size
        self.patch_size = np.round(anchor_size * (image_size - 1)).astype(int)
        self.margin = margin
        self.crop_size = self.patch_size + 2 * margin
        self.ch = ch
        self.enable_attn = enable_attn
        self.use_resblock = use_resblock
        self.features_dim = features_dim
        self.hidden_dims = hidden_dims
        hidden_dim_1 = hidden_dims[0]
        hidden_dim_2 = hidden_dims[1]

        in_ch = ch
        self.cnn = KeyPointCNNOriginal(cdim=in_ch, channels=cnn_channels, image_size=self.crop_size, n_kp=32,
                                       pad_mode='replicate', use_resblock=self.use_resblock,
                                       first_conv_kernel_size=3)

        feature_map_size = (self.crop_size // 2 ** (len(cnn_channels) - 1)) ** 2
        # fc_in_dim = 32 * ((self.crop_size // 4) ** 2) if cnn else self.ch * (self.crop_size ** 2)
        fc_in_dim = 32 * feature_map_size
        # self.backbone = nn.Sequential(nn.Linear(fc_in_dim, hidden_dim_1),
        #                               nn.ReLU(True),
        #                               nn.Linear(hidden_dim_1, hidden_dim_2),
        #                               nn.ReLU(True))
        # self.projection = nn.Linear(fc_in_dim, hidden_dim_2)
        # attention
        if enable_attn:
            self.projection = nn.Linear(fc_in_dim, hidden_dim_2)
            # self.projection = nn.Sequential(nn.LayerNorm(fc_in_dim),
            #                                 nn.Linear(fc_in_dim, hidden_dim_2),
            #                                 nn.LayerNorm(hidden_dim_2))
            self.pos_enc = CoordinateEncoding(encoded_size=hidden_dim_2 // 2, is_random=True, is_learned=False)
            self.pos_emb = nn.Sequential(nn.Linear(hidden_dim_2 + 2, 4 * hidden_dim_2),
                                         nn.GELU(),
                                         nn.Linear(4 * hidden_dim_2, hidden_dim_2)
                                         )
            self.self_attention = AttentionBlock(n_embed=hidden_dim_2, n_head=1, resid_pdrop=attn_dropout,
                                                 attn_pdrop=attn_dropout,
                                                 layer_norm=True)
        else:
            # self.projection = nn.Identity()
            self.projection = nn.Linear(fc_in_dim, hidden_dim_2)
            self.pos_enc = nn.Identity()
            self.pos_emb = nn.Identity()
            self.self_attention = nn.Identity()

        self.backbone = nn.Sequential(nn.Linear(hidden_dim_2, hidden_dim_1),
                                      nn.ReLU(True),
                                      nn.Linear(hidden_dim_1, hidden_dim_2),
                                      nn.ReLU(True))
        self.features_head = nn.Linear(hidden_dim_2, 2 * self.features_dim)  # mu_features, logvar_features

    def forward(self, x, kp, z_scale=None):
        # x: [bs, ch, image_size, image_size]
        # kp: [bs, n_kp, 2] in [-1, 1]
        batch_size = x.shape[0]
        n_kp = kp.shape[1]
        img_size = x.shape[-1]
        x_repeated = x.unsqueeze(1).repeat(1, n_kp, 1, 1, 1)  # [batch_size, n_kp, ch, image_size, image_size]
        x_repeated = x_repeated.view(-1, *x.shape[1:])  # [batch_size * n_kp, ch, image_size, image_size]
        if z_scale is None:
            z_scale = (self.patch_size / img_size) * torch.ones_like(kp)
        else:
            # assume unnormalized z_scale
            z_scale = torch.sigmoid(z_scale)
        z_pos = kp.reshape(-1, kp.shape[-1])
        z_scale = z_scale.view(-1, z_scale.shape[-1])
        out_dims = (batch_size * n_kp, x.shape[1], self.patch_size, self.patch_size)
        cropped_objects = spatial_transform(x_repeated, z_pos, z_scale, out_dims, inverse=False)
        # [batch_size * n_kp, ch, patch_size, patch_size]

        # encode objects - fc
        _, cropped_objects_cnn = self.cnn(cropped_objects)
        cropped_objects_flat = cropped_objects_cnn.reshape(batch_size, n_kp, -1)  # flatten
        # backbone features
        # backbone_features = self.backbone(cropped_objects_flat)
        backbone_features = cropped_objects_flat
        # projection
        backbone_features = self.projection(backbone_features)

        # attention
        if self.enable_attn:
            # projection
            # backbone_features = self.projection(backbone_features)
            # positional encoding
            kp_pos_enc = self.pos_emb(torch.cat([self.pos_enc(kp.detach()), kp.detach()], dim=-1))
            backbone_features_in = backbone_features + kp_pos_enc  # [batch_size, n_kp, hidden_dim_2]
            # attention + residual
            # backbone_features = backbone_features + self.self_attention(backbone_features_in)
            backbone_features = self.self_attention(backbone_features_in)

        backbone_features = self.backbone(backbone_features)
        # projection to output

        features = self.features_head(backbone_features)
        features = features.view(batch_size, n_kp, -1)
        mu_features, logvar_features = torch.chunk(features, 2, dim=-1)

        cropped_objects = cropped_objects.view(batch_size, -1, *cropped_objects.shape[1:])
        # [batch_size, n_kp, ch, crop_size, crop_size]
        spatial_out = {'mu_features': mu_features, 'logvar_features': logvar_features,
                       'cropped_objects': cropped_objects}
        return spatial_out


class ObjectDecoderCNN(nn.Module):
    def __init__(self, patch_size, num_chans=4, bottleneck_size=128, pad_mode='reflect', embed_position=False,
                 use_resblock=False):
        super().__init__()

        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.num_chans = num_chans
        self.embed_position = embed_position
        self.use_resblock = use_resblock
        if self.embed_position:
            self.position_embedding = nn.Linear(2, bottleneck_size)

        self.in_ch = 32

        fc_out_dim = self.in_ch * 8 * 8
        fc_in_dim = bottleneck_size if not self.embed_position else 2 * bottleneck_size
        self.fc = nn.Sequential(nn.Linear(fc_in_dim, 256, bias=True),
                                nn.ReLU(True),
                                nn.Linear(256, fc_out_dim),
                                nn.ReLU(True))

        num_upsample = int(np.log(patch_size[0]) // np.log(2)) - 3
        print(f'ObjDecCNN: fc to cnn num upsample: {num_upsample}')
        self.channels = [self.in_ch]
        for i in range(num_upsample):
            self.channels.append(64)
        # self.channels = (32, 64, 64)
        cc = self.channels[-1]

        sz = 8

        self.main = nn.Sequential()
        if num_upsample > 0:
            self.main.add_module('depth_up',
                                 ConvBlock(self.in_ch, cc, kernel_size=3, pad=1, upsample=True, pad_mode=pad_mode,
                                           use_resblock=self.use_resblock))
            for ch in reversed(self.channels[1:-1]):
                self.main.add_module('conv_to_{}'.format(sz * 2), ConvBlock(cc, ch, kernel_size=3, pad=1, upsample=True,
                                                                            pad_mode=pad_mode,
                                                                            use_resblock=self.use_resblock))
                cc, sz = ch, sz * 2

        self.main.add_module('conv_to_{}'.format(sz * 2),
                             ConvBlock(cc, self.channels[0], kernel_size=3, pad=1,
                                       upsample=False, pad_mode=pad_mode, use_resblock=False))
        self.main.add_module('final_conv', ConvBlock(self.channels[0], num_chans, kernel_size=1, bias=True,
                                                     activation=False, batchnorm=False, use_resblock=False))
        self.decode = self.main

    def forward(self, x, kp=None):
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        if kp is not None and self.embed_position:
            pos_embed = self.position_embedding(kp)  # [bs, n_kp, bottleneck_size]
            pos_embed = pos_embed.view(-1, pos_embed.shape[-1])
            x = torch.cat([x, pos_embed], dim=-1)
        conv_in = self.fc(x)
        conv_in = conv_in.view(-1, 32, 8, 8)
        out = self.decode(conv_in).view(-1, self.num_chans, *self.patch_size)
        out = torch.sigmoid(out)
        return out


class FCToCNN(nn.Module):
    def __init__(self, target_hw=16, n_ch=8, pad_mode='replicate', features_dim=2, use_resblock=False):
        super(FCToCNN, self).__init__()
        # features_dim : 2 [logvar] + additional features
        self.features_dim = features_dim  # logvar, features
        self.n_ch = n_ch
        self.fmap_size = 8
        self.use_resblock = use_resblock
        fc_out_dim = self.n_ch * (self.fmap_size ** 2)

        self.mlp = nn.Sequential(nn.Linear(self.features_dim, 64),
                                 nn.ReLU(True),
                                 nn.Linear(64, 128),
                                 nn.ReLU(True),
                                 nn.Linear(128, 256),
                                 nn.ReLU(True),
                                 nn.Linear(256, 512),
                                 nn.ReLU(True),
                                 nn.Linear(512, fc_out_dim),
                                 nn.ReLU(True))

        num_upsample = int(np.log(target_hw) // np.log(2)) - 3
        # print(f'pointnet to cnn num upsample: {num_upsample}')
        self.cnn = nn.Sequential()
        for i in range(num_upsample):
            self.cnn.add_module(f'depth_up_{i}', ConvBlock(n_ch, n_ch, kernel_size=3, pad=1,
                                                           upsample=True, pad_mode=pad_mode,
                                                           use_resblock=self.use_resblock))

    def forward(self, features):
        # features [batch_size, features_dim]
        features = features.view(-1, features.shape[-1])  # [batch_size * n_kp, features]
        h = self.mlp(features)
        h = h.view(-1, self.n_ch, self.fmap_size, self.fmap_size)  # [batch_size, n_kp, 4, 4]
        cnn_out = self.cnn(h)  # [batch_size, n_kp, target_hw, target_hw]
        return cnn_out
