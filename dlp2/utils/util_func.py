# imports
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import datetime
import os
import json
import imageio
# torch
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.ops as ops
from typing import Tuple


def color_map(num=100):
    colormap = ["FF355E",
                "8ffe09",
                "1d5dec",
                "FF9933",
                "FFFF66",
                "CCFF00",
                "AAF0D1",
                "FF6EFF",
                "FF00CC",
                "299617",
                "AF6E4D"] * num
    s = ''
    for color in colormap:
        s += color
    b = bytes.fromhex(s)
    cm = np.frombuffer(b, np.uint8)
    cm = cm.reshape(len(colormap), 3)
    return cm


def plot_keypoints_on_image(k, image_tensor, radius=1, thickness=1, kp_range=(0, 1), plot_numbers=True):
    # https://github.com/DuaneNielsen/keypoints
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    text_color = (255, 255, 255)
    # Line thickness of 2 px
    # thickness = 2
    height, width = image_tensor.size(1), image_tensor.size(2)
    num_keypoints = k.size(0)

    if len(k.shape) != 2:
        raise Exception('Individual media and keypoints, not batches')

    k = k.clone()
    k[:, 0] = ((k[:, 0] - kp_range[0]) / (kp_range[1] - kp_range[0])) * (height - 1)
    k[:, 1] = ((k[:, 1] - kp_range[0]) / (kp_range[1] - kp_range[0])) * (width - 1)
    # k.floor_()
    k.round_()
    k = k.detach().cpu().numpy()
    #     print(k)

    img = transforms.ToPILImage()(image_tensor.cpu())

    img = np.array(img)
    cmap = color_map()
    cm = cmap[:num_keypoints].astype(int)
    count = 0
    eps = 8
    for co_ord, color in zip(k, cm):
        c = color.item(0), color.item(1), color.item(2)
        co_ord = co_ord.squeeze()
        cv2.circle(img, (int(co_ord[1]), int(co_ord[0])), radius, c, thickness)
        if plot_numbers:
            cv2.putText(img, f'{count}', (int(co_ord[1] - eps), int(co_ord[0] - eps)), font, fontScale, c, 2,
                        cv2.LINE_AA)
        count += 1
    #     print(f'{count} kp plotted')

    # img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    # img = np.transpose(img, (2, 0, 1))
    # img = Image.fromarray(img)

    return img


def plot_keypoints_on_image_batch(kp_batch_tensor, img_batch_tensor, radius=1, thickness=1, max_imgs=8,
                                  kp_range=(0, 1), plot_numbers=False):
    num_plot = min(max_imgs, img_batch_tensor.shape[0])
    img_with_kp = []
    for i in range(num_plot):
        img_np = plot_keypoints_on_image(kp_batch_tensor[i], img_batch_tensor[i], radius=radius, thickness=1,
                                         kp_range=kp_range, plot_numbers=plot_numbers)
        img_tensor = torch.tensor(img_np).float() / 255.0
        img_with_kp.append(img_tensor.permute(2, 0, 1))
    img_with_kp = torch.stack(img_with_kp, dim=0)
    return img_with_kp


def plot_batch_kp(img_batch_tensor, kp_batch_tensor, rec_batch_tensor, max_imgs=8):
    batch_size, _, _, im_size = img_batch_tensor.shape
    max_index = (im_size - 1)
    #     print(max_index)
    num_plot = min(max_imgs, img_batch_tensor.shape[0])
    img_with_kp_np = []
    for i in range(num_plot):
        img_with_kp_np.append(plot_keypoints_on_image(kp_batch_tensor[i], img_batch_tensor[i], radius=2, thickness=1))
    img_np = img_batch_tensor.permute(0, 2, 3, 1).clamp(0, 1).data.cpu().numpy()
    rec_np = rec_batch_tensor.permute(0, 2, 3, 1).clamp(0, 1).data.cpu().numpy()
    fig = plt.figure()
    for i in range(num_plot):
        # image
        ax = fig.add_subplot(3, num_plot, i + 1)
        ax.imshow(img_np[i])
        ax.axis('equal')
        ax.set_axis_off()
        # kp
        ax = fig.add_subplot(3, num_plot, i + 1 + num_plot)
        ax.imshow(img_with_kp_np[i])
        ax.axis('equal')
        ax.set_axis_off()
        # rec
        ax = fig.add_subplot(3, num_plot, i + 1 + 2 * num_plot)
        ax.imshow(rec_np[i])
        ax.axis('equal')
        ax.set_axis_off()
    return fig


def reparameterize(mu, logvar):
    """
    This function applies the reparameterization trick:
    z = mu(X) + sigma(X)^0.5 * epsilon, where epsilon ~ N(0,I)
    :param mu: mean of x
    :param logvar: log variaance of x
    :return z: the sampled latent variable
    """
    device = mu.device
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(mu).to(device)
    return mu + eps * std


def create_masks_fast(center, anchor_s, feature_dim=16, patch_size=None):
    # center: [batch_size, n_kp, 2] in kp_range
    # anchor_h, anchor_w: size of anchor in [0, 1]
    batch_size, n_kp = center.shape[0], center.shape[1]
    if patch_size is None:
        patch_size = np.round(anchor_s * (feature_dim - 1)).astype(int)
    # create white rectangles
    masks = torch.ones(batch_size * n_kp, 1, patch_size, patch_size, device=center.device).float()
    # pad the masks to image size
    pad_size = (feature_dim - patch_size) // 2
    padded_patches_batch = F.pad(masks, pad=[pad_size] * 4)
    # move the masks to be centered around the kp
    delta_t_batch = 0.0 - center
    delta_t_batch = delta_t_batch.reshape(-1, delta_t_batch.shape[-1])  # [bs * n_kp, 2]
    zeros = torch.zeros([delta_t_batch.shape[0], 1], device=delta_t_batch.device).float()
    ones = torch.ones([delta_t_batch.shape[0], 1], device=delta_t_batch.device).float()
    theta = torch.cat([ones, zeros, delta_t_batch[:, 1].unsqueeze(-1),
                       zeros, ones, delta_t_batch[:, 0].unsqueeze(-1)], dim=-1)
    theta = theta.view(-1, 2, 3)  # [batch_size * n_kp, 2, 3]
    align_corners = False
    padding_mode = 'zeros'
    mode = "nearest"
    # mode = 'bilinear'

    trans_padded_patches_batch = affine_grid_sample(padded_patches_batch, theta, padded_patches_batch.shape, mode=mode)

    # grid = F.affine_grid(theta, padded_patches_batch.size(), align_corners=align_corners)
    # trans_padded_patches_batch = F.grid_sample(padded_patches_batch, grid, align_corners=align_corners,
    #                                            mode=mode, padding_mode=padding_mode)

    trans_padded_patches_batch = trans_padded_patches_batch.view(batch_size, n_kp, *padded_patches_batch.shape[1:])
    # [bs, n_kp, 1, feature_dim, feature_dim]
    return trans_padded_patches_batch


def get_bb_from_masks(masks, width, height):
    # extracts bounding boxes (bb) from masks.
    # batch version
    # https://discuss.pytorch.org/t/find-bounding-box-around-ones-in-batch-of-masks/141266
    # masks: [n_masks, 1, feature_dim, feature_dim]
    masks = masks.bool().squeeze(1)
    b, h, w = masks.shape
    coor = torch.zeros(size=(b, 4), dtype=torch.int, device=masks.device)
    scales = torch.zeros(size=(b, 2), dtype=torch.float, device=masks.device)  # normalized scales
    centers = torch.zeros(size=(b, 2), dtype=torch.float, device=masks.device)  # normalized [-1, 1] centers of bbs

    rows = torch.any(masks, axis=2)
    cols = torch.any(masks, axis=1)

    rmins = torch.argmax(rows.float(), dim=1)
    rmaxs = h - torch.argmax(rows.float().flip(dims=[1]), dim=1) - 1
    cmins = torch.argmax(cols.float(), dim=1)
    cmaxs = w - torch.argmax(cols.float().flip(dims=[1]), dim=1) - 1

    ws = (cmins * (width / w)).clamp(0, width).int()
    wt = (cmaxs * (width / w)).clamp(0, width).int()
    hs = (rmins * (height / h)).clamp(0, height).int()
    ht = (rmaxs * (height / h)).clamp(0, height).int()

    coor[:, 0] = ws  # ws
    coor[:, 1] = hs  # hs
    coor[:, 2] = wt  # wt
    coor[:, 3] = ht  # ht

    # normalized scales
    scales[:, 1] = (wt - ws) / width
    scales[:, 0] = (ht - hs) / height
    # normalized centers
    centers[:, 1] = 2 * (((ws + wt) / 2) / width - 0.5)
    centers[:, 0] = 2 * (((hs + ht) / 2) / height - 0.5)

    output_dict = {'coor': coor, 'scales': scales, 'centers': centers}

    return output_dict


def get_bb_from_z_scale(kp, z_scale, width, height, scale_normalized=False):
    # extracts bounding boxes (bb) from keypoints and scales.
    # kp: [n_kp, 2], range: (-1, 1)
    # z_scale: [n_kp, 2], range: (0, 1)
    # scale_normalized: False if scale is not in [0, 1]
    n_kp = kp.shape[0]
    coor = torch.zeros(size=(n_kp, 4), dtype=torch.int, device=kp.device)
    kp_norm = 0.5 + kp / 2  # [0, 1]
    if scale_normalized:
        scale_norm = z_scale
    else:
        # scale_norm = 0.5 + z_scale / 2
        scale_norm = torch.sigmoid(z_scale)
    for i in range(n_kp):
        x_kp = kp_norm[i, 1] * width
        x_scale = scale_norm[i, 1] * width
        y_kp = kp_norm[i, 0] * height
        y_scale = scale_norm[i, 0] * height
        ws = (x_kp - x_scale / 2).clamp(0, width).int()
        wt = (x_kp + x_scale / 2).clamp(0, width).int()
        hs = (y_kp - y_scale / 2).clamp(0, height).int()
        ht = (y_kp + y_scale / 2).clamp(0, height).int()
        coor[i, 0] = ws
        coor[i, 1] = hs
        coor[i, 2] = wt
        coor[i, 3] = ht
    return coor


def get_bb_from_masks_batch(masks, width, height):
    # extracts bounding boxes (bb) from a batch of masks.
    # masks: [batch_size, n_masks, 1, feature_dim, feature_dim]
    coor = torch.zeros(size=(masks.shape[0], masks.shape[1], 4), dtype=torch.int, device=masks.device)
    for i in range(masks.shape[0]):
        coor[i, :, :] = get_bb_from_masks(masks[i], width, height)
    return coor


def nms_single(boxes, scores, iou_thresh=0.5, return_scores=False, remove_ind=None):
    # non-maximal suppression on bb and scores from one image.
    # boxes: [n_bb, 4], scores: [n_boxes]
    nms_indices = ops.nms(boxes.float(), scores, iou_thresh)
    # remove low scoring indices from nms output
    if remove_ind is not None:
        # final_indices = [ind for ind in nms_indices if ind not in remove_ind]
        final_indices = list(set(nms_indices.data.cpu().numpy()) - set(remove_ind))
        # print(f'removed indices: {remove_ind}')
    else:
        final_indices = nms_indices
    nms_boxes = boxes[final_indices]  # [n_bb_nms, 4]
    if return_scores:
        return nms_boxes, final_indices, scores[final_indices]
    else:
        return nms_boxes, final_indices


def remove_low_score_bb_single(boxes, scores, return_scores=False, mode='mean', thresh=0.4, hard_thresh=None):
    # filters out low-scoring bounding boxes. The score is usually the variance of the particle.
    # boxes: [n_bb, 4], scores: [n_boxes]
    if hard_thresh is None:
        if mode == 'mean':
            mean_score = scores.mean()
            # indices = (scores > mean_score)
            indices = torch.nonzero(scores > thresh, as_tuple=True)[0].data.cpu().numpy()
        else:
            normalzied_scores = (scores - scores.min()) / (scores.max() - scores.min())
            # indices = (normalzied_scores > thresh)
            indices = torch.nonzero(normalzied_scores > thresh, as_tuple=True)[0].data.cpu().numpy()
    else:
        # indices = (scores > hard_thresh)
        indices = torch.nonzero(scores > hard_thresh, as_tuple=True)[0].data.cpu().numpy()
    boxes_t = boxes[indices]
    scores_t = scores[indices]
    if return_scores:
        return indices, boxes_t, scores_t
    else:
        return indices, boxes_t


def get_low_score_bb_single(scores, mode='mean', thresh=0.4, hard_thresh=None):
    # get indices of low-scoring bounding boxes.
    # boxes: [n_bb, 4], scores: [n_boxes]
    if hard_thresh is None:
        if mode == 'mean':
            mean_score = scores.mean()
            # indices = (scores > mean_score)
            indices = torch.nonzero(scores < thresh, as_tuple=True)[0].data.cpu().numpy()
        else:
            normalzied_scores = (scores - scores.min()) / (scores.max() - scores.min())
            # indices = (normalzied_scores > thresh)
            indices = torch.nonzero(normalzied_scores < thresh, as_tuple=True)[0].data.cpu().numpy()
    else:
        # indices = (scores > hard_thresh)
        indices = torch.nonzero(scores < hard_thresh, as_tuple=True)[0].data.cpu().numpy()
    return indices


def plot_bb_on_image_from_masks_nms(masks, image_tensor, scores, iou_thresh=0.5, thickness=1, hard_thresh=None):
    # plot bounding boxes on a single image, use non-maximal suppression to filter low-scoring bbs.
    # masks: [n_masks, 1, feature_dim, feature_dim]
    n_masks = masks.shape[0]
    mask_h, mask_w = masks.shape[2], masks.shape[3]
    height, width = image_tensor.size(1), image_tensor.size(2)
    img = transforms.ToPILImage()(image_tensor.cpu())
    img = np.array(img)
    cmap = color_map()
    cm = cmap[:n_masks].astype(int)
    count = 0
    # get bb coor
    bb_from_masks = get_bb_from_masks(masks, width, height)
    coors = bb_from_masks['coor']  # [n_masks, 4]
    # remove low-score bb
    if hard_thresh is None:
        low_score_ind = get_low_score_bb_single(scores, mode='mean', hard_thresh=2.0)
    else:
        low_score_ind = get_low_score_bb_single(scores, mode='mean', hard_thresh=hard_thresh)
    # nms
    # remove_ind = low_score_ind if hard_thresh is not None else None
    remove_ind = low_score_ind
    # nms
    # remove_ind = low_score_ind if hard_thresh is not None else None
    coors_nms, nms_indices, scores_nms = nms_single(coors, scores, iou_thresh, return_scores=True,
                                                    remove_ind=remove_ind)
    # [n_masks_nms, 4]
    for coor, color in zip(coors_nms, cm):
        c = color.item(0), color.item(1), color.item(2)
        ws = (coor[0] - thickness).clamp(0, width)
        hs = (coor[1] - thickness).clamp(0, height)
        wt = (coor[2] + thickness).clamp(0, width)
        ht = (coor[3] + thickness).clamp(0, height)
        bb_s = (int(ws), int(hs))
        bb_t = (int(wt), int(ht))
        cv2.rectangle(img, bb_s, bb_t, c, thickness, 1)
        score_text = f'{scores_nms[count]:.2f}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.3
        thickness = 1
        box_w = bb_t[0] - bb_s[0]
        box_h = bb_t[1] - bb_s[1]
        org = (int(bb_s[0] + box_w / 4), int(bb_s[1] + box_h / 2))
        cv2.putText(img, score_text, org, font, fontScale, thickness=thickness, color=c, lineType=cv2.LINE_AA)
        count += 1

    return img, nms_indices


def plot_bb_on_image_batch_from_masks_nms(mask_batch_tensor, img_batch_tensor, scores, iou_thresh=0.5, thickness=1,
                                          max_imgs=8, hard_thresh=None):
    # plot bounding boxes on a batch of media, use non-maximal suppression to filter low-scoring bbs.
    # mask_batch_tensor: [batch_size, n_kp, 1, feature_dim, feature_dim]
    num_plot = min(max_imgs, img_batch_tensor.shape[0])
    img_with_bb = []
    indices = []
    for i in range(num_plot):
        img_np, nms_indices = plot_bb_on_image_from_masks_nms(mask_batch_tensor[i], img_batch_tensor[i], scores[i],
                                                              iou_thresh, thickness=thickness, hard_thresh=hard_thresh)
        img_tensor = torch.tensor(img_np).float() / 255.0
        img_with_bb.append(img_tensor.permute(2, 0, 1))
        indices.append(nms_indices)
    img_with_bb = torch.stack(img_with_bb, dim=0)
    return img_with_bb, indices


def plot_bb_on_image_from_z_scale_nms(kp, z_scale, image_tensor, scores, iou_thresh=0.5, thickness=1, hard_thresh=None,
                                      scale_normalized=False):
    # plot bounding boxes on a single image, use non-maximal suppression to filter low-scoring bbs.
    # kp: [n_kp, 2], range: (-1, 1)
    # z_scale: [n_kp, 2], range: (0, 1)
    n_kp = kp.shape[0]
    height, width = image_tensor.size(1), image_tensor.size(2)
    img = transforms.ToPILImage()(image_tensor.cpu())
    img = np.array(img)
    cmap = color_map()
    cm = cmap[:n_kp].astype(int)
    count = 0
    # get bb coor
    coors = get_bb_from_z_scale(kp, z_scale, width, height, scale_normalized=scale_normalized)  # [n_masks, 4]
    # remove low-score bb
    if hard_thresh is None:
        low_score_ind = get_low_score_bb_single(scores, mode='mean', hard_thresh=2.0)
    else:
        low_score_ind = get_low_score_bb_single(scores, mode='mean', hard_thresh=hard_thresh)
    # nms
    # remove_ind = low_score_ind if hard_thresh is not None else None
    remove_ind = low_score_ind
    coors_nms, nms_indices, scores_nms = nms_single(coors, scores, iou_thresh, return_scores=True,
                                                    remove_ind=remove_ind)
    # [n_masks_nms, 4]
    for coor, color in zip(coors_nms, cm):
        c = color.item(0), color.item(1), color.item(2)
        ws = (coor[0] - thickness).clamp(0, width)
        hs = (coor[1] - thickness).clamp(0, height)
        wt = (coor[2] + thickness).clamp(0, width)
        ht = (coor[3] + thickness).clamp(0, height)
        bb_s = (int(ws), int(hs))
        bb_t = (int(wt), int(ht))
        cv2.rectangle(img, bb_s, bb_t, c, thickness, 1)
        score_text = f'{scores_nms[count]:.2f}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.3
        thickness = 1
        box_w = bb_t[0] - bb_s[0]
        box_h = bb_t[1] - bb_s[1]
        org = (int(bb_s[0] + box_w / 4), int(bb_s[1] + box_h / 2))
        cv2.putText(img, score_text, org, font, fontScale, thickness=thickness, color=c, lineType=cv2.LINE_AA)
        count += 1

    return img, nms_indices


def plot_bb_on_image_batch_from_z_scale_nms(kp_batch_tensor, z_scale_batch_tensor, img_batch_tensor, scores,
                                            iou_thresh=0.5, thickness=1, max_imgs=8, hard_thresh=None,
                                            scale_normalized=False):
    # plot bounding boxes on a batch of media, use non-maximal suppression to filter low-scoring bbs.
    # kp_batch_tensor: [batch_size, n_kp, 2]
    # z_scale_batch_tensor: [batch_size, n_kp, 2]
    num_plot = min(max_imgs, img_batch_tensor.shape[0])
    img_with_bb = []
    indices = []
    for i in range(num_plot):
        img_np, nms_indices = plot_bb_on_image_from_z_scale_nms(kp_batch_tensor[i], z_scale_batch_tensor[i],
                                                                img_batch_tensor[i], scores[i], iou_thresh,
                                                                thickness=thickness, hard_thresh=hard_thresh,
                                                                scale_normalized=scale_normalized)
        img_tensor = torch.tensor(img_np).float() / 255.0
        img_with_bb.append(img_tensor.permute(2, 0, 1))
        indices.append(nms_indices)
    img_with_bb = torch.stack(img_with_bb, dim=0)
    return img_with_bb, indices


def plot_bb_on_image_from_masks(masks, image_tensor, thickness=1):
    # vanilla plotting of bbs from masks.
    # masks: [n_masks, 1, feature_dim, feature_dim]
    n_masks = masks.shape[0]
    mask_h, mask_w = masks.shape[2], masks.shape[3]
    height, width = image_tensor.size(1), image_tensor.size(2)

    img = transforms.ToPILImage()(image_tensor.cpu())

    img = np.array(img)
    cmap = color_map()
    cm = cmap[:n_masks].astype(int)
    count = 0
    for mask, color in zip(masks, cm):
        c = color.item(0), color.item(1), color.item(2)
        mask = mask.int().squeeze()  # [feature_dim, feature_dim]
        #         print(mask.shape)
        indices = (mask == 1).nonzero(as_tuple=False)
        #         print(indices.shape)
        if indices.shape[0] > 0:
            ws = (indices[0][1] * (width / mask_w) - thickness).clamp(0, width).int()
            wt = (indices[-1][1] * (width / mask_w) + thickness).clamp(0, width).int()
            hs = (indices[0][0] * (height / mask_h) - thickness).clamp(0, height).int()
            ht = (indices[-1][0] * (height / mask_h) + thickness).clamp(0, height).int()
            bb_s = (int(ws), int(hs))
            bb_t = (int(wt), int(ht))
            cv2.rectangle(img, bb_s, bb_t, c, thickness, 1)
            count += 1
    return img


def plot_bb_on_image_batch_from_masks(mask_batch_tensor, img_batch_tensor, thickness=1, max_imgs=8):
    # vanilla plotting of bbs from a batch of masks.
    # mask_batch_tensor: [batch_size, n_kp, 1, feature_dim, feature_dim]
    num_plot = min(max_imgs, img_batch_tensor.shape[0])
    img_with_bb = []
    for i in range(num_plot):
        img_np = plot_bb_on_image_from_masks(mask_batch_tensor[i], img_batch_tensor[i], thickness=thickness)
        img_tensor = torch.tensor(img_np).float() / 255.0
        img_with_bb.append(img_tensor.permute(2, 0, 1))
    img_with_bb = torch.stack(img_with_bb, dim=0)
    return img_with_bb


def plot_glimpse_obj_on(dec_object_glimpses, obj_on, save_dir):
    _, dec_object_glimpses = torch.split(dec_object_glimpses, [1, 3], dim=2)
    B, N, C, H, W = dec_object_glimpses.shape
    n_row, n_col = 1, B

    fig = plt.figure(figsize=(2 * n_col, 7 * n_row))
    fig.suptitle(f"Particle Glimpses Object-On", fontsize=20)

    for i in range(B):
        ax = fig.add_subplot(n_row, n_col, i+1)
        glimpses = dec_object_glimpses[i]
        glimpses = torch.cat([glimpses[i] for i in range(len(glimpses))], dim=1)
        glimpses = glimpses.detach().cpu().numpy()
        glimpses = np.moveaxis(glimpses, 0, -1)
        ax.imshow(glimpses)
        ax.set_xticks([], [])
        # ax.set_yticks(range(W // 2 - 1, W // 2 + W * n_particles - 1, W), ['*'] + [f"{i}" for i in range(1, n_particles)])
        ax.set_yticks(range(W // 2 - 1, W // 2 + W * N - 1, W), [f"{obj_on[i][m]:1.2f}" for m in range(N)])
        for j in range(1, N):
            ax.axhline(y=j * W, color='black')

    fig.tight_layout()
    # plt.show()
    plt.savefig(save_dir, bbox_inches='tight')
    return

def farthest_point_sample(x, n_sample, rand_init=False, return_idx=False):
    """
    Adapted from: https://github.com/DylanWusee/pointconv_pytorch/blob/master/utils/pointconv_util.py#L60
    Input:
        x: pointcloud data, [batch_size, n_points, n_dim]
        n_sample: number of samples
    Return:
        sampled_points: sampled points, [batch_size, n_sample, n_dim]
        centroids: sampled pointcloud indices, [batch_size, n_sample]
    """
    device = x.device
    batch_size, n_points, n_dim = x.shape
    centroids = torch.zeros(batch_size, n_sample, dtype=torch.long, device=device)
    distance = torch.ones(batch_size, n_points, device=device) * 1e10
    if rand_init:
        farthest = torch.randint(0, n_points, (batch_size,), dtype=torch.long, device=device)
    else:
        # farthest = torch.zeros(batch_size, dtype=torch.long, device=device)
        # mid init
        farthest = int(n_points // 2) * torch.ones(batch_size, dtype=torch.long, device=device)
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)
    for i in range(n_sample):
        centroids[:, i] = farthest
        centroid = x[batch_indices, farthest, :].view(batch_size, 1, n_dim)
        dist = torch.sum((x - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]  # get the index of the farthest point
    sampled_points = x[batch_indices[:, None], centroids]  # [batch_size, n_sample, n_dim]
    if return_idx:
        return sampled_points, centroids
    else:
        return sampled_points


def prepare_logdir(runname, src_dir='./', accelerator=None):
    td_prefix = datetime.datetime.now().strftime("%d%m%y_%H%M%S")
    dir_name = f'{td_prefix}_{runname}'
    path_to_dir = os.path.join(src_dir, dir_name)
    path_to_fig_dir = os.path.join(path_to_dir, 'figures')
    path_to_save_dir = os.path.join(path_to_dir, 'saves')
    if accelerator is not None and accelerator.is_main_process:
        os.makedirs(path_to_dir, exist_ok=True)
        os.makedirs(path_to_fig_dir, exist_ok=True)
        os.makedirs(path_to_save_dir, exist_ok=True)
    elif accelerator is None:
        os.makedirs(path_to_dir, exist_ok=True)
        os.makedirs(path_to_fig_dir, exist_ok=True)
        os.makedirs(path_to_save_dir, exist_ok=True)
    else:
        pass
    return path_to_dir


def save_config(src_dir, hparams):
    path_to_conf = os.path.join(src_dir, 'hparams.json')
    with open(path_to_conf, "w") as outfile:
        json.dump(hparams, outfile, indent=2)


def log_line(src_dir, line):
    log_file = os.path.join(src_dir, 'log.txt')
    with open(log_file, 'a') as fp:
        fp.writelines(line)


def animate_trajectories(orig_trajectory, pred_trajectory, path='./traj_anim.gif', duration=4 / 50, rec_to_pred_t=10):
    # rec_to_pred_t: the timestep from which prediction transitions from reconstruction to generation
    # prepare media
    font = cv2.FONT_HERSHEY_SIMPLEX
    origin = (5, 15)
    fontScale = 0.4
    color = (255, 255, 255)
    gt_border_color = (255, 0, 0)
    rec_border_color = (0, 0, 255)
    gen_border_color = (0, 255, 0)
    border_size = 2
    thickness = 1
    gt_traj_prep = []
    pred_traj_prep = []
    for i in range(orig_trajectory.shape[0]):
        image = (orig_trajectory[i] * 255).astype(np.uint8).copy()
        image = cv2.putText(image, f'GT:{i}', origin, font, fontScale, color, thickness, cv2.LINE_AA)
        # add border
        image = cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT,
                                   value=gt_border_color)
        gt_traj_prep.append(image)

        text = f'REC:{i}' if i < rec_to_pred_t else f'PRED:{i}'
        image = (pred_trajectory[i].clip(0, 1) * 255).astype(np.uint8).copy()
        image = cv2.putText(image, text, origin, font, fontScale, color, thickness, cv2.LINE_AA)
        # add border
        border_color = rec_border_color if i < rec_to_pred_t else gen_border_color
        image = cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT,
                                   value=border_color)
        pred_traj_prep.append(image)

    total_images = []
    for i in range(len(orig_trajectory)):
        white_border = (np.ones((gt_traj_prep[i].shape[0], 4, gt_traj_prep[i].shape[-1])) * 255).astype(np.uint8)
        concat_img = np.concatenate([gt_traj_prep[i],
                                     white_border,
                                     pred_traj_prep[i]], axis=1)
        # total_images.append((concat_img * 255).astype(np.uint8))
        total_images.append(concat_img)
    imageio.mimsave(path, total_images, duration=duration)  # 1/50


def spatial_transform(image, z_pos, z_scale, out_dims, inverse=False, eps=1e-9):
    """
    https://github.com/zhixuan-lin/G-SWM
    spatial transformer network used to scale and shift input according to z_where in:
            1/ x -> x_att   -- shapes (H, W) -> (attn_window, attn_window) -- thus inverse = False
            2/ y_att -> y   -- (attn_window, attn_window) -> (H, W) -- thus inverse = True
    inverting the affine transform as follows: A_inv ( A * image ) = image
    A = [R | T] where R is rotation component of angle alpha, T is [tx, ty] translation component
    A_inv rotates by -alpha and translates by [-tx, -ty]
    if x' = R * x + T  -->  x = R_inv * (x' - T) = R_inv * x - R_inv * T
    here, z_where is 3-dim [scale, tx, ty] so inverse transform is [1/scale, -tx/scale, -ty/scale]
    R = [[s, 0],  ->  R_inv = [[1/s, 0],
         [0, s]]               [0, 1/s]]
    ------
    image: [batch_size * n_kp, ch, h, w]
    z_pos: [batch_size * n_kp, 2]
    z_scale: [batch_size * n_kp, 2]
    out_dims: tuple (batch_size * n_kp, ch, h*, w*)
    """
    # 1. construct 2x3 affine matrix for each datapoint in the batch
    theta = torch.zeros(2, 3, device=image.device).repeat(image.shape[0], 1, 1)
    # set scaling
    theta[:, 0, 0] = z_scale[:, 1] if not inverse else 1 / (z_scale[:, 1] + eps)
    theta[:, 1, 1] = z_scale[:, 0] if not inverse else 1 / (z_scale[:, 0] + eps)

    # set translation
    theta[:, 0, -1] = z_pos[:, 1] if not inverse else - z_pos[:, 1] / (z_scale[:, 1] + eps)
    theta[:, 1, -1] = z_pos[:, 0] if not inverse else - z_pos[:, 0] / (z_scale[:, 0] + eps)
    # construct sampling grid and sample image from grid
    return affine_grid_sample(image, theta, out_dims, mode='bilinear')

    # # 2. construct sampling grid
    # grid = F.affine_grid(theta, torch.Size(out_dims), align_corners=True)
    # # 3. sample image from grid
    # mode = 'bilinear'
    # #     mode = 'nearest'
    # return F.grid_sample(image, grid, align_corners=True, mode=mode)


# learning rate warmup + cosine scheduler
class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    # https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


def generate_correlation_maps(x, kp, patch_size, previous_objects=None, z_scale=None):
    """
    Generates correlation heatmaps between patches of size `patch_size` extracted from `kp` and `x`, where the template
    to match is given by patches from the previous timestep `previous_objects`.
    x: [batch_size, ch, h, w] in [0, 1]
    kp: [batch_size, n_kp, 2] in [-1, 1]
    z_scale: [batch_size, n_kp, 2] in [-1, 1]
    previous_objects: [bs * n_kp, 3, patch_size, patch_size] (the template to match)
    returns [bs * n_kp, 4, patch_size, patch_size]
    """
    pad_size = patch_size
    batch_size = x.shape[0]
    img_size = x.shape[-1]
    n_kp = kp.shape[1]
    pad_func = torch.nn.ReplicationPad2d(pad_size)

    x_repeat = x.unsqueeze(1).repeat(1, n_kp, 1, 1, 1)  # [bs, n_kp, ch, h, w]
    x_repeat = x_repeat.view(-1, *x_repeat.shape[2:])  # [bs * n_kp, ch, h, w]
    # extract patches
    if z_scale is None:
        z_scale = (patch_size / img_size) * torch.ones_like(kp)
    else:
        # assume z_scale is not normalized, need to bring to [0, 1]
        z_scale = torch.sigmoid(z_scale)
    z_pos = kp.reshape(-1, kp.shape[-1])  # [bs * n_kp, 2]
    z_scale = z_scale.view(-1, z_scale.shape[-1])
    out_dims = (batch_size * n_kp, x.shape[1], patch_size, patch_size)
    cropped_objects = spatial_transform(x_repeat, z_pos, z_scale, out_dims, inverse=False)
    # cropped_objects: [bs * n_kp, ch, patch_size, patch_size]

    if previous_objects is None:
        # do nothing
        cropped_heatmaps = torch.ones(cropped_objects.shape[0], 1, patch_size, patch_size,
                                      device=cropped_objects.device)
    else:
        in0 = cropped_objects.reshape(1, -1, *cropped_objects.shape[2:])
        # [1, bs * n_kp * ch, patch_size, patch_size]
        in0 = pad_func(in0)
        # print(f'previous_objects: {previous_objects.shape}')
        in1 = previous_objects.reshape(-1, *previous_objects.shape[1:])  # [bs * n_kp, ch, patch_size, patch_size]
        output = correlate(in0, in1)
        # reshape to original
        output = output.view(batch_size * n_kp, -1, *output.shape[2:])
        # output = F.interpolate(output, size=(in0.shape[-2], in0.shape[-1]), mode='bilinear', align_corners=True)
        output = output[:, :, pad_size // 2 + 1:-pad_size // 2, pad_size // 2 + 1:-pad_size // 2]
        # output: [bs * n_kp, 1, patch_size, patch_size]
        # out_dims = (batch_size * n_kp, output.shape[1], patch_size, patch_size)
        # cropped_heatmaps = spatial_transform(output, z_pos, z_scale, out_dims, inverse=False)
        cropped_heatmaps = output
    # cropped_heatmaps: [bs * n_kp, 1, patch_size, patch_size]
    return torch.cat([cropped_objects, cropped_heatmaps], dim=1)  # [bs * n_kp, 4, patch_size, patch_size]


def calc_gradient_norm(model):
    total_norm = 0
    max_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        if param_norm.item() > max_norm:
            max_norm = param_norm.item()
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm, max_norm


"""
JIT scripts
"""


@torch.jit.script
def correlate(x, kernel):
    groups = kernel.shape[0]
    output = F.conv2d(x, kernel, padding=0, groups=groups, stride=1, bias=None)
    norm = torch.sqrt(torch.sum(kernel ** 2) * F.conv2d(x ** 2, torch.ones_like(kernel), groups=groups,
                                                        bias=None, stride=1, padding=0))
    output = output / (norm + 1e-5)
    return output


@torch.jit.script
def affine_grid_sample(x, theta, out_dims: Tuple[int, int, int, int], mode: str):
    # construct sampling grid
    grid = F.affine_grid(theta, torch.Size(out_dims), align_corners=True)
    # sample image from grid
    return F.grid_sample(x, grid, align_corners=True, mode=mode)
