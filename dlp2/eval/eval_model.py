"""
Evaluation of the ELBO on the validation set
"""
# imports
import numpy as np
import os
# torch
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.utils as vutils

# datasets
from dlp2.datasets.panda_ds import PandaPush
# util functions
from dlp2.utils.loss_functions import calc_reconstruction_loss, VGGDistance
from dlp2.utils.util_func import plot_keypoints_on_image_batch, \
    plot_bb_on_image_batch_from_z_scale_nms, plot_bb_on_image_batch_from_masks_nms, plot_glimpse_obj_on


def evaluate_validation_elbo(model, ds, data_root_dir, epoch, batch_size=100, recon_loss_type="vgg", device=torch.device('cpu'),
                             save_image=False, fig_dir='./', topk=5, recon_loss_func=None, beta_rec=1.0, beta_kl=1.0,
                             kl_balance=1.0, accelerator=None, model_type='object', iou_thresh=0.2, ):
    model.eval()
    kp_range = model.kp_range
    # load data
    if ds == "panda_push":
        image_size = 128
        dataset = PandaPush(data_root_dir, mode='valid', res=image_size)
    else:
        raise NotImplementedError

    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=2, drop_last=False)
    if recon_loss_func is None:
        if recon_loss_type == "vgg":
            recon_loss_func = VGGDistance(device=device)
        else:
            recon_loss_func = calc_reconstruction_loss

    elbos = []
    for batch in dataloader:
        if ds == 'panda_push':
            x = batch[0].squeeze(1).to(device)
            x_prior = x
        else:
            x = batch
            x_prior = x
        batch_size = x.shape[0]
        # forward pass
        with torch.no_grad():
            model_output = model(x, x_prior=x_prior)

        mu_p = model_output['kp_p']
        # gmap = model_output['gmap']
        mu = model_output['mu']
        logvar = model_output['logvar']
        z_base = model_output['z_base']
        mu_offset = model_output['mu_offset']
        logvar_offset = model_output['logvar_offset']
        rec_x = model_output['rec']
        mu_features = model_output['mu_features']
        logvar_features = model_output['logvar_features']
        mu_scale = model_output['mu_scale']
        logvar_scale = model_output['logvar_scale']
        mu_depth = model_output['mu_depth']
        logvar_depth = model_output['logvar_depth']
        # object stuff
        dec_objects_original = model_output['dec_objects_original']
        cropped_objects_original = model_output['cropped_objects_original']
        obj_on = model_output['obj_on']  # [batch_size, n_kp]
        obj_on_a = model_output['obj_on_a']  # [batch_size, n_kp]
        obj_on_b = model_output['obj_on_b']  # [batch_size, n_kp]
        alpha_masks = model_output['alpha_masks']  # [batch_size, n_kp, 1, h, w]

        all_losses = model.calc_elbo(x, model_output, beta_kl=beta_kl,
                                     beta_rec=beta_rec, kl_balance=kl_balance,
                                     recon_loss_type=recon_loss_type,
                                     recon_loss_func=recon_loss_func)
        loss = all_losses['loss']

        # for plotting, confidence calculation
        mu_tot = z_base + mu_offset
        logvar_tot = logvar_offset

        elbo = loss
        elbos.append(elbo.data.cpu().numpy())
    if save_image:
        max_imgs = 8
        mu_plot = mu_tot.clamp(min=kp_range[0], max=kp_range[1])
        img_with_kp = plot_keypoints_on_image_batch(mu_plot, x, radius=3,
                                                    thickness=1, max_imgs=max_imgs, kp_range=model.kp_range)
        img_with_kp_p = plot_keypoints_on_image_batch(mu_p, x_prior, radius=3, thickness=1, max_imgs=max_imgs,
                                                      kp_range=model.kp_range)
        # top-k
        with torch.no_grad():
            # logvar_sum = logvar[:, :-1].sum(-1) * obj_on  # [bs, n_kp]
            logvar_sum = logvar_tot.sum(-1) * obj_on  # [bs, n_kp]
            logvar_topk = torch.topk(logvar_sum, k=topk, dim=-1, largest=False)
            indices = logvar_topk[1]  # [batch_size, topk]
            # batch_indices = torch.arange(mu.shape[0]).view(-1, 1).to(mu.device)
            batch_indices = torch.arange(mu_tot.shape[0]).view(-1, 1).to(mu_tot.device)
            # topk_kp = mu[batch_indices, indices]
            topk_kp = mu_tot[batch_indices, indices]
            # bounding boxes
            bb_scores = -1 * logvar_sum
            # hard_threshold = bb_scores.mean()
            hard_threshold = None
            # kp_batch = mu[:, :-1].clamp(min=kp_range[0], max=kp_range[1])
            kp_batch = mu_plot
            scale_batch = mu_scale
            img_with_masks_nms, nms_ind = plot_bb_on_image_batch_from_z_scale_nms(kp_batch, scale_batch, x,
                                                                                  scores=bb_scores,
                                                                                  iou_thresh=iou_thresh,
                                                                                  thickness=1,
                                                                                  max_imgs=max_imgs,
                                                                                  hard_thresh=hard_threshold)
            alpha_masks = torch.where(alpha_masks < 0.05, 0.0, 1.0)
            img_with_masks_alpha_nms, _ = plot_bb_on_image_batch_from_masks_nms(alpha_masks, x,
                                                                                scores=bb_scores,
                                                                                iou_thresh=iou_thresh,
                                                                                thickness=1,
                                                                                max_imgs=max_imgs,
                                                                                hard_thresh=hard_threshold)
        img_with_kp_topk = plot_keypoints_on_image_batch(topk_kp.clamp(min=kp_range[0], max=kp_range[1]), x,
                                                         radius=3, thickness=1, max_imgs=max_imgs,
                                                         kp_range=kp_range)
        if model_type == 'object' and dec_objects_original is not None:
            dec_objects = model_output['dec_objects']
            bg = model_output['bg']
            if accelerator is not None:
                if accelerator.is_main_process:
                    vutils.save_image(torch.cat([x[:max_imgs, -3:], img_with_kp[:max_imgs, -3:].to(accelerator.device),
                                                 rec_x[:max_imgs, -3:],
                                                 img_with_kp_p[:max_imgs, -3:].to(accelerator.device),
                                                 img_with_kp_topk[:max_imgs, -3:].to(accelerator.device),
                                                 dec_objects[:max_imgs, -3:],
                                                 img_with_masks_nms[:max_imgs, -3:].to(accelerator.device),
                                                 img_with_masks_alpha_nms[:max_imgs, -3:].to(accelerator.device),
                                                 bg[:max_imgs, -3:]],
                                                dim=0).data.cpu(), '{}/image_valid_{}.jpg'.format(fig_dir, epoch),
                                      nrow=8, pad_value=1)
            else:
                vutils.save_image(torch.cat([x[:max_imgs, -3:], img_with_kp[:max_imgs, -3:].to(mu.device),
                                             rec_x[:max_imgs, -3:],
                                             img_with_kp_p[:max_imgs, -3:].to(mu.device),
                                             img_with_kp_topk[:max_imgs, -3:].to(mu.device),
                                             dec_objects[:max_imgs, -3:],
                                             img_with_masks_nms[:max_imgs, -3:].to(mu.device),
                                             img_with_masks_alpha_nms[:max_imgs, -3:].to(mu.device),
                                             bg[:max_imgs, -3:]],
                                            dim=0).data.cpu(), '{}/image_valid_{}.jpg'.format(fig_dir, epoch),
                                  nrow=8, pad_value=1)
            with torch.no_grad():
                _, dec_objects_rgb = torch.split(dec_objects_original, [1, 3], dim=2)
                dec_objects_rgb = dec_objects_rgb.reshape(-1, *dec_objects_rgb.shape[2:])
                cropped_objects_original = cropped_objects_original.clone().reshape(-1, 3,
                                                                                    cropped_objects_original.shape[
                                                                                        -1],
                                                                                    cropped_objects_original.shape[
                                                                                        -1])
                if cropped_objects_original.shape[-1] != dec_objects_rgb.shape[-1]:
                    cropped_objects_original = F.interpolate(cropped_objects_original,
                                                             size=dec_objects_rgb.shape[-1],
                                                             align_corners=False, mode='bilinear')
            if accelerator is not None:
                if accelerator.is_main_process:
                    vutils.save_image(
                        torch.cat([cropped_objects_original[:max_imgs * 2, -3:], dec_objects_rgb[:max_imgs * 2, -3:]],
                                  dim=0).data.cpu(), '{}/image_obj_valid_{}.jpg'.format(fig_dir, epoch),
                        nrow=8, pad_value=1)
            else:
                vutils.save_image(
                    torch.cat([cropped_objects_original[:max_imgs * 2, -3:], dec_objects_rgb[:max_imgs * 2, -3:]],
                              dim=0).data.cpu(), '{}/image_obj_valid_{}.jpg'.format(fig_dir, epoch),
                    nrow=8, pad_value=1)
        else:
            if accelerator is not None:
                if accelerator.is_main_process:
                    vutils.save_image(torch.cat([x[:max_imgs, -3:], img_with_kp[:max_imgs, -3:].to(mu.device),
                                                 rec_x[:max_imgs, -3:], img_with_kp_p[:max_imgs, -3:].to(mu.device),
                                                 img_with_kp_topk[:max_imgs, -3:].to(mu.device)],
                                                dim=0).data.cpu(), '{}/image_valid_{}.jpg'.format(fig_dir, epoch),
                                      nrow=8, pad_value=1)
            else:
                vutils.save_image(torch.cat([x[:max_imgs, -3:], img_with_kp[:max_imgs, -3:].to(mu.device),
                                             rec_x[:max_imgs, -3:], img_with_kp_p[:max_imgs, -3:].to(mu.device),
                                             img_with_kp_topk[:max_imgs, -3:].to(mu.device)],
                                            dim=0).data.cpu(), '{}/image_valid_{}.jpg'.format(fig_dir, epoch),
                                  nrow=8, pad_value=1)

        # save obj_on image
        plot_glimpse_obj_on(model_output['dec_objects_original'][:max_imgs], obj_on[:max_imgs], '{}/obj_on_image_valid_{}.jpg'.format(fig_dir, epoch))

    return np.mean(elbos)
