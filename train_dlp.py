"""
Main training function for single-GPU machines
Default hyper-parameters
+---------+--------------------------------+----------+-------------+---------------+---------+------------+------------+----------+---------------------+
| dataset | model (decoder_type) | n_kp_enc | n_kp_prior  | rec_loss_func | beta_kl | kl_balance | patch_size | anchor_s | learned_feature_dim |
+---------+--------------------------------+----------+-------------+---------------+---------+------------+------------+----------+---------------------+
| celeb   | masked (masked)      |       30 |          50 | vgg           |      40 |      0.001 |          8 |    0.125 |                  10 |
| traffic | object (object)      |       15 |          20 | vgg           |      30 |      0.001 |         16 |     0.25 |                  20 |
| clevrer | object (object)      |       10 |          20 | vgg           |      40 |      0.001 |         16 |     0.25 |                   5 |
| shapes  | object (object)      |        8 |          15 | mse           |    0.1  |      0.001 |          8 |     0.25 |                   5 |
+---------+--------------------------------+----------+-------------+---------------+---------+------------+------------+----------+---------------------+
"""
# imports
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib
import yaml
from pathlib import Path
# torch
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torch.optim as optim
# modules
from dlp2.models import ObjectDLP
# datasets
from dlp2.datasets.panda_ds import PandaPush

# util functions
from dlp2.utils.loss_functions import calc_reconstruction_loss, VGGDistance
from dlp2.utils.util_func import plot_keypoints_on_image_batch, prepare_logdir, save_config, log_line, \
    plot_bb_on_image_batch_from_z_scale_nms, plot_bb_on_image_batch_from_masks_nms, plot_glimpse_obj_on
from dlp2.eval.eval_model import evaluate_validation_elbo
from dlp2.eval.eval_gen_metrics import eval_im_metric

matplotlib.use("Agg")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def train_dlp(ds, data_root_dir, batch_size=16, lr=2e-4, device=torch.device("cpu"), kp_activation="tanh",
              pad_mode='replicate', num_epochs=250, load_model=False, n_kp=8, recon_loss_type="mse",
              sigma=1.0, beta_kl=1.0, beta_rec=1.0, dropout=0.0,
              patch_size=16, topk=15, n_kp_enc=20, eval_epoch_freq=5,
              learned_feature_dim=5, bg_learned_feature_dim=5, n_kp_prior=100, weight_decay=0.0, kp_range=(-1, 1),
              run_prefix="", warmup_epoch=5, iou_thresh=0.15, anchor_s=0.25,
              kl_balance=0.001, scale_std=0.3, offset_std=0.2,
              obj_on_alpha=0.1, obj_on_beta=0.1, eval_im_metrics=False, use_correlation_heatmaps=False):
    """
    ds: dataset name (str)
    data_root_dir: dataset root directory (str)
    enc_channels: channels for the posterior CNN (takes in the whole image)
    prior_channels: channels for prior CNN (takes in patches)
    n_kp: number of kp to extract from each (!) patch
    n_kp_prior: number of kp to filter from the set of prior kp (of size n_kp x num_patches)
    n_kp_enc: number of posterior kp to be learned (this is the actual number of kp that will be learnt)
    pad_mode: padding for the CNNs, 'zeros' or  'replicate' (default)
    sigma: the prior std of the KP
    dropout: dropout for the CNNs. We don't use it though...
    patch_size: patch size for the prior KP proposals network (not to be confused with the glimpse size)
    kp_range: the range of keypoints, can be [-1, 1] (default) or [0,1]
    learned_feature_dim: the latent visual features dimensions extracted from glimpses.
    kp_activation: the type of activation to apply on the keypoints: "tanh" for kp_range [-1, 1], "sigmoid" for [0, 1]
    anchor_s: defines the glimpse size as a ratio of image_size (e.g., 0.25 for image_size=128 -> glimpse_size=32)
    iou_thresh: intersection-over-union threshold for non-maximal suppression (nms) to filter bounding boxes
    topk: the number top-k particles with the lowest variance (highest confidence) to filter for the plots.
    warmup_epoch: (used for the Object Model) number of epochs where only the object decoder is trained.
    recon_loss_type: tpe of pixel reconstruction loss ("mse", "vgg").
    beta_rec: coefficient for the reconstruction loss (we use 1.0).
    beta_kl: coefficient for the KL divergence term in the loss.
    kl_balance: coefficient for the balance between the ChamferKL (for the KP)
                and the standard KL (for the visual features),
                kl_loss = beta_kl * (chamfer_kl + kl_balance * kl_features)
    scale_std: prior std for the scale
    offset_std: prior std for the offset
    obj_on_alpha: prior alpha (Beta distribution) for obj_on
    obj_on_beta: prior beta (Beta distribution) for obj_on
    eval_im_metric: evaluate LPIPS, SSIM, PSNR during training
    use_correlation_heatmaps: calculate correlation maps between patches for tracking
    """

    # load data
    if ds == "panda_push":
        image_size = 128
        ch = 3
        enc_channels = [32, 64, 128]
        prior_channels = (16, 32, 64)
        dataset = PandaPush(data_root_dir, mode='train', res=image_size)
        milestones = (20, 40, 80)
    else:
        raise NotImplementedError

    # save hyper-parameters
    hparams = {'ds': ds, 'batch_size': batch_size, 'lr': lr, 'kp_activation': kp_activation, 'pad_mode': pad_mode,
               'num_epochs': num_epochs, 'n_kp': n_kp, 'recon_loss_type': recon_loss_type,
               'sigma': sigma, 'beta_kl': beta_kl, 'beta_rec': beta_rec,
               'patch_size': patch_size, 'topk': topk, 'n_kp_enc': n_kp_enc,
               'eval_epoch_freq': eval_epoch_freq, 'learned_feature_dim': learned_feature_dim,
               'bg_learned_feature_dim': bg_learned_feature_dim,
               'n_kp_prior': n_kp_prior, 'weight_decay': weight_decay, 'kp_range': kp_range,
               'run_prefix': run_prefix,
               'warmup_epoch': warmup_epoch,
               'iou_thresh': iou_thresh, 'anchor_s': anchor_s, 'kl_balance': kl_balance,
               'milestones': milestones, 'image_size': image_size, 'cdim': ch, 'enc_channels': enc_channels,
               'prior_channels': prior_channels,
               'scale_std': scale_std, 'offset_std': offset_std, 'obj_on_alpha': obj_on_alpha,
               'obj_on_beta': obj_on_beta, 'use_correlation_heatmaps': use_correlation_heatmaps}

    # create dataloader
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True,
                            drop_last=True)
    # model
    model = ObjectDLP(cdim=ch, enc_channels=enc_channels, prior_channels=prior_channels,
                      image_size=image_size, n_kp=n_kp, learned_feature_dim=learned_feature_dim,
                      bg_learned_feature_dim=bg_learned_feature_dim, pad_mode=pad_mode, sigma=sigma,
                      dropout=dropout, patch_size=patch_size, n_kp_enc=n_kp_enc,
                      n_kp_prior=n_kp_prior, kp_range=kp_range, kp_activation=kp_activation,
                      anchor_s=anchor_s, use_resblock=False,
                      scale_std=scale_std,
                      offset_std=offset_std, obj_on_alpha=obj_on_alpha,
                      obj_on_beta=obj_on_beta, use_correlation_heatmaps=use_correlation_heatmaps).to(device)

    # prior logvars
    # prepare saving location
    run_name = f'{ds}_dlp' + run_prefix
    log_dir = prepare_logdir(runname=run_name, src_dir='dlp2/')
    fig_dir = os.path.join(log_dir, 'figures')
    save_dir = os.path.join(log_dir, 'saves')
    save_config(log_dir, hparams)

    if recon_loss_type == "vgg":
        recon_loss_func = VGGDistance(device=device)
    else:
        recon_loss_func = calc_reconstruction_loss
    betas = (0.9, 0.999)
    eps = 1e-4
    optimizer = optim.Adam(model.get_parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95, verbose=True)

    if load_model:
        try:
            model.load_state_dict(
                torch.load(os.path.join(save_dir, f'{ds}_dlp.pth'), map_location=device))
            print("loaded model from checkpoint")
        except:
            print("model checkpoint not found")

    # statistics
    losses = []
    losses_rec = []
    losses_kl = []
    losses_kl_kp = []
    losses_kl_feat = []

    # image metrics
    if eval_im_metrics:
        val_lpipss = []
        best_val_lpips_epoch = 0
        val_lpips = best_val_lpips = 1e8

    # initialize validation statistics
    valid_loss = best_valid_loss = 1e8
    valid_losses = []
    best_valid_epoch = 0

    # save PSNR values of the reconstruction
    psnrs = []

    for epoch in range(num_epochs):
        model.train()
        batch_losses = []
        batch_losses_rec = []
        batch_losses_kl = []
        batch_losses_kl_kp = []
        batch_losses_kl_feat = []
        batch_psnrs = []
        pbar = tqdm(iterable=dataloader)
        for batch in pbar:
            if ds == 'panda_push':
                x = batch[0].squeeze(1).to(device)
                x_prior = x
            else:
                x = batch.to(device)
                x_prior = x
            batch_size = x.shape[0]
            # forward pass
            noisy = (epoch < (warmup_epoch + 1))  # add small noise to the alpha masks
            train_enc_prior = True
            model_output = model(x, x_prior=x_prior, warmup=(epoch < warmup_epoch), noisy=noisy,
                                 train_prior=train_enc_prior)
            mu_p = model_output['kp_p']
            mu = model_output['mu']
            logvar = model_output['logvar']
            z_base = model_output['z_base']
            z = model_output['z']
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

            # calc elbo
            all_losses = model.calc_elbo(x, model_output, warmup=(epoch < warmup_epoch), beta_kl=beta_kl,
                                         beta_rec=beta_rec, kl_balance=kl_balance,
                                         recon_loss_type=recon_loss_type,
                                         recon_loss_func=recon_loss_func)
            psnr = all_losses['psnr']
            obj_on_l1 = all_losses['obj_on_l1']
            loss = all_losses['loss']
            loss_kl = all_losses['kl']
            loss_rec = all_losses['loss_rec']
            loss_kl_kp = all_losses['loss_kl_kp']
            loss_kl_feat = all_losses['loss_kl_feat']
            loss_kl_obj_on = all_losses['loss_kl_obj_on']
            loss_kl_scale = all_losses['loss_kl_scale']
            loss_kl_depth = all_losses['loss_kl_depth']

            # for plotting, confidence calculation
            mu_tot = z_base + mu_offset
            logvar_tot = logvar_offset

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # log
            batch_psnrs.append(psnr.data.cpu().item())
            batch_losses.append(loss.data.cpu().item())
            batch_losses_rec.append(loss_rec.data.cpu().item())
            batch_losses_kl.append(loss_kl.data.cpu().item())
            batch_losses_kl_kp.append(loss_kl_kp.data.cpu().item())
            batch_losses_kl_feat.append(loss_kl_feat.data.cpu().item())
            # progress bar
            if epoch < warmup_epoch:
                pbar.set_description_str(f'epoch #{epoch} (warmup)')
            elif noisy:
                pbar.set_description_str(f'epoch #{epoch} (noisy)')
            else:
                pbar.set_description_str(f'epoch #{epoch}')

            pbar.set_postfix(loss=loss.data.cpu().item(), rec=loss_rec.data.cpu().item(),
                             kl=loss_kl.data.cpu().item(), on_l1=obj_on_l1.cpu().item())
            # break  # for testing
        pbar.close()
        losses.append(np.mean(batch_losses))
        losses_rec.append(np.mean(batch_losses_rec))
        losses_kl.append(np.mean(batch_losses_kl))
        losses_kl_kp.append(np.mean(batch_losses_kl_kp))
        losses_kl_feat.append(np.mean(batch_losses_kl_feat))
        if len(batch_psnrs) > 0:
            psnrs.append(np.mean(batch_psnrs))
        # schedulers
        scheduler.step()
        # epoch summary
        log_str = f'epoch {epoch} summary\n'
        log_str += f'loss: {losses[-1]:.3f}, rec: {losses_rec[-1]:.3f}, kl: {losses_kl[-1]:.3f}\n'
        log_str += f'kl_balance: {kl_balance:.4f}, kl_kp: {losses_kl_kp[-1]:.3f}, kl_feat: {losses_kl_feat[-1]:.3f}\n'
        log_str += f'mu max: {mu.max()}, mu min: {mu.min()}\n'
        log_str += f'mu offset max: {mu_offset.max()}, mu offset min: {mu_offset.min()}\n'
        log_str += f'val loss (freq: {eval_epoch_freq}): {valid_loss:.3f},' \
                   f' best: {best_valid_loss:.3f} @ epoch: {best_valid_epoch}\n'
        if obj_on is not None:
            log_str += f'obj_on max: {obj_on.max()}, obj_on min: {obj_on.min()}\n'
            # log_str += f'scale max: {mu_scale.max()}, scale min: {mu_scale.min()}\n'
            log_str += f'scale max: {mu_scale.sigmoid().max()}, scale min: {mu_scale.sigmoid().min()}\n'
            log_str += f'depth max: {mu_depth.max()}, depth min: {mu_depth.min()}\n'
        if eval_im_metrics:
            log_str += f'val lpips (freq: {eval_epoch_freq}): {val_lpips:.3f},' \
                       f' best: {best_val_lpips:.3f} @ epoch: {best_val_lpips_epoch}\n'
        if len(psnrs) > 0:
            log_str += f'mean psnr: {psnrs[-1]:.3f}\n'
        print(log_str)
        log_line(log_dir, log_str)

        if epoch % eval_epoch_freq == 0 or epoch == num_epochs - 1:
            # for plotting purposes
            mu_plot = mu_tot.clamp(min=kp_range[0], max=kp_range[1])
            # mu_plot = (mu[:, :-1]).clamp(min=kp_range[0], max=kp_range[1])
            max_imgs = 8
            img_with_kp = plot_keypoints_on_image_batch(mu_plot, x, radius=3,
                                                        thickness=1, max_imgs=max_imgs, kp_range=kp_range)
            img_with_kp_p = plot_keypoints_on_image_batch(mu_p, x_prior, radius=3, thickness=1, max_imgs=max_imgs,
                                                          kp_range=kp_range)
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
                                                                                  thickness=1, max_imgs=max_imgs,
                                                                                  hard_thresh=hard_threshold)
            alpha_masks = torch.where(alpha_masks < 0.05, 0.0, 1.0)
            img_with_masks_alpha_nms, _ = plot_bb_on_image_batch_from_masks_nms(alpha_masks, x, scores=bb_scores,
                                                                                iou_thresh=iou_thresh, thickness=1,
                                                                                max_imgs=max_imgs,
                                                                                hard_thresh=hard_threshold)
            # hard_thresh: a general threshold for bb scores (set None to not use it)
            bb_str = f'bb scores: max: {bb_scores.max():.2f}, min: {bb_scores.min():.2f},' \
                     f' mean: {bb_scores.mean():.2f}\n'
            print(bb_str)
            log_line(log_dir, bb_str)
            img_with_kp_topk = plot_keypoints_on_image_batch(topk_kp.clamp(min=kp_range[0], max=kp_range[1]), x,
                                                             radius=3, thickness=1, max_imgs=max_imgs,
                                                             kp_range=kp_range)
            if dec_objects_original is not None:
                dec_objects = model_output['dec_objects']
                bg = model_output['bg']
                vutils.save_image(torch.cat([x[:max_imgs, -3:], img_with_kp[:max_imgs, -3:].to(device),
                                             rec_x[:max_imgs, -3:], img_with_kp_p[:max_imgs, -3:].to(device),
                                             img_with_kp_topk[:max_imgs, -3:].to(device),
                                             dec_objects[:max_imgs, -3:],
                                             img_with_masks_nms[:max_imgs, -3:].to(device),
                                             img_with_masks_alpha_nms[:max_imgs, -3:].to(device),
                                             bg[:max_imgs, -3:]],
                                            dim=0).data.cpu(), '{}/image_{}.jpg'.format(fig_dir, epoch),
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
                vutils.save_image(
                    torch.cat([cropped_objects_original[:max_imgs * 2, -3:], dec_objects_rgb[:max_imgs * 2, -3:]],
                              dim=0).data.cpu(), '{}/image_obj_{}.jpg'.format(fig_dir, epoch),
                    nrow=8, pad_value=1)
            else:
                vutils.save_image(torch.cat([x[:max_imgs, -3:], img_with_kp[:max_imgs, -3:].to(device),
                                             rec_x[:max_imgs, -3:], img_with_kp_p[:max_imgs, -3:].to(device),
                                             img_with_kp_topk[:max_imgs, -3:].to(device)],
                                            dim=0).data.cpu(), '{}/image_{}.jpg'.format(fig_dir, epoch),
                                  nrow=8, pad_value=1)
            # save obj_on image
            plot_glimpse_obj_on(model_output['dec_objects_original'][:max_imgs], obj_on[:max_imgs], '{}/obj_on_image_{}.jpg'.format(fig_dir, epoch))
            # save model
            torch.save(model.state_dict(),
                       os.path.join(save_dir, f'{ds}_dlp{run_prefix}.pth'))
            print(f'validation step...')
            valid_loss = evaluate_validation_elbo(model, ds, data_root_dir, epoch, batch_size=batch_size,
                                                  recon_loss_type=recon_loss_type, device=device,
                                                  save_image=True, fig_dir=fig_dir, topk=topk,
                                                  recon_loss_func=recon_loss_func, beta_rec=beta_rec,
                                                  beta_kl=beta_kl, kl_balance=kl_balance)
            log_str = f'validation loss: {valid_loss:.3f}\n'
            print(log_str)
            log_line(log_dir, log_str)
            if best_valid_loss > valid_loss:
                log_str = f'validation loss updated: {best_valid_loss:.3f} -> {valid_loss:.3f}\n'
                print(log_str)
                log_line(log_dir, log_str)
                best_valid_loss = valid_loss
                best_valid_epoch = epoch
                torch.save(model.state_dict(),
                           os.path.join(save_dir,
                                        f'{ds}_dlp{run_prefix}_best.pth'))
            torch.cuda.empty_cache()
            if eval_im_metrics and epoch > 0:
                valid_imm_results = eval_im_metric(model, device, data_root_dir,
                                                   val_mode='val',
                                                   ds=ds,
                                                   eval_dir=log_dir,
                                                   batch_size=batch_size)
                log_str = f'validation: lpips: {valid_imm_results["lpips"]:.3f}, '
                log_str += f'psnr: {valid_imm_results["psnr"]:.3f}, ssim: {valid_imm_results["ssim"]:.3f}\n'
                val_lpips = valid_imm_results['lpips']
                print(log_str)
                log_line(log_dir, log_str)
                if (not torch.isinf(torch.tensor(val_lpips))) and (best_val_lpips > val_lpips):
                    log_str = f'validation lpips updated: {best_val_lpips:.3f} -> {val_lpips:.3f}\n'
                    print(log_str)
                    log_line(log_dir, log_str)
                    best_val_lpips = val_lpips
                    best_val_lpips_epoch = epoch
                    torch.save(model.state_dict(),
                               os.path.join(save_dir, f'{ds}_dlp{run_prefix}_best_lpips.pth'))
                torch.cuda.empty_cache()
        valid_losses.append(valid_loss)
        if eval_im_metrics:
            val_lpipss.append(val_lpips)
        # plot graphs
        if epoch > 0:
            num_plots = 4
            fig = plt.figure()
            ax = fig.add_subplot(num_plots, 1, 1)
            ax.plot(np.arange(len(losses[1:])), losses[1:], label="loss")
            ax.set_title(run_name)
            ax.legend()

            ax = fig.add_subplot(num_plots, 1, 2)
            ax.plot(np.arange(len(losses_kl[1:])), losses_kl[1:], label="kl", color='red')
            if learned_feature_dim > 0:
                ax.plot(np.arange(len(losses_kl_kp[1:])), losses_kl_kp[1:], label="kl_kp", color='cyan')
                ax.plot(np.arange(len(losses_kl_feat[1:])), losses_kl_feat[1:], label="kl_feat", color='green')
            ax.legend()

            ax = fig.add_subplot(num_plots, 1, 3)
            ax.plot(np.arange(len(losses_rec[1:])), losses_rec[1:], label="rec", color='green')
            ax.legend()

            ax = fig.add_subplot(num_plots, 1, 4)
            ax.plot(np.arange(len(valid_losses[1:])), valid_losses[1:], label="valid_loss", color='magenta')
            ax.legend()
            plt.tight_layout()
            plt.savefig(f'{fig_dir}/{run_name}_graph.jpg')
            plt.close('all')
    return model


if __name__ == "__main__":

    config = yaml.safe_load(Path('config/TrainDLPConfig.yaml').read_text())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    run_prefix = f"_{config['n_kp_enc']}kp_{config['n_kp_prior']}kpp_{config['learned_feature_dim']}zdim"

    model = train_dlp(ds=config['ds'], data_root_dir=config['data_root_dir'], run_prefix=run_prefix,
                      device=device, load_model=config['load_model'],
                      lr=config['lr'], batch_size=config['batch_size'],
                      weight_decay=config['weight_decay'], dropout=config['dropout'],
                      num_epochs=config['num_epochs'], warmup_epoch=config['warmup_epoch'],
                      kp_activation=config['kp_activation'], kp_range=config['kp_range'],
                      n_kp=config['n_kp'], n_kp_enc=config['n_kp_enc'], n_kp_prior=config['n_kp_prior'],
                      pad_mode=config['pad_mode'], sigma=config['sigma'], patch_size=config['patch_size'],
                      beta_kl=config['beta_kl'], beta_rec=config['beta_rec'], kl_balance=config['kl_balance'],
                      learned_feature_dim=config['learned_feature_dim'],
                      bg_learned_feature_dim=config['bg_learned_feature_dim'],
                      recon_loss_type=config['recon_loss_type'], topk=config['topk'],
                      anchor_s=config['anchor_s'], scale_std=config['scale_std'], offset_std=config['offset_std'],
                      eval_epoch_freq=config['eval_epoch_freq'], eval_im_metrics=config['eval_im_metrics'])
