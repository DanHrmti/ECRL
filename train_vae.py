"""
Single-GPU training of VAE
"""
# imports
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib
# torch
import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torch.optim as optim
# modules
from vae.models.get_model import get_model
# datasets
from vae.datasets.get_dataset import get_image_dataset
# util functions
from vae.utils import prepare_logdir, save_config, log_line, get_config
from vae.eval.eval_metrics import eval_im_metric, evaluate_validation_loss

matplotlib.use("Agg")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def train_vae(config_path='./configs/panda.json'):
    # load config
    try:
        config = get_config(config_path)
    except FileNotFoundError:
        raise SystemExit("config file not found")
    hparams = config  # to save a copy of the hyper-parameters
    # data and general
    model_type = config['model_type']
    ds = config['ds']
    ch = config['ch']  # image channels
    image_size = config['image_size']
    root = config['root']  # dataset root
    batch_size = config['batch_size']
    lr = config['lr']
    num_epochs = config['num_epochs']
    eval_epoch_freq = config['eval_epoch_freq']
    weight_decay = config['weight_decay']
    run_prefix = config['run_prefix']
    load_model = config['load_model']
    pretrained_path = config['pretrained_path']  # path of pretrained model to load, if None, train from scratch
    adam_betas = config['adam_betas']
    adam_eps = config['adam_eps']
    scheduler_gamma = config['scheduler_gamma']
    eval_im_metrics = config['eval_im_metrics']
    device = config['device']
    if 'cuda' in device:
        device = torch.device(f'{device}' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    # optimization
    warmup_epoch = config['warmup_epoch']
    recon_loss_type = config['recon_loss_type']
    beta_kl = config['beta_kl']
    beta_rec = config['beta_rec']

    # load data
    dataset = get_image_dataset(ds, root, mode='train', image_size=image_size)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True,
                            drop_last=True)
    # model
    model = get_model(config).to(device)
    print(model.info())
    # prepare saving location
    run_name = f'{ds}_{model_type}' + run_prefix
    log_dir = prepare_logdir(runname=run_name, src_dir='vae/')
    fig_dir = os.path.join(log_dir, 'figures')
    save_dir = os.path.join(log_dir, 'saves')
    save_config(log_dir, hparams)

    # optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=adam_betas, eps=adam_eps, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=scheduler_gamma, verbose=True)

    if load_model and pretrained_path is not None:
        try:
            model.load_state_dict(torch.load(pretrained_path, map_location=device))
            print("loaded model from checkpoint")
        except:
            print("model checkpoint not found")

    # log statistics
    losses = []
    losses_rec = []
    losses_kl = []

    # initialize validation statistics
    valid_loss = best_valid_loss = 1e8
    valid_losses = []
    best_valid_epoch = 0

    # image metrics
    if eval_im_metrics:
        val_lpipss = []
        best_val_lpips_epoch = 0
        val_lpips = best_val_lpips = 1e8

    for epoch in range(num_epochs):
        model.train()
        batch_losses = []
        batch_losses_rec = []
        batch_losses_kl = []

        pbar = tqdm(iterable=dataloader)
        for batch in pbar:
            x = batch[0].to(device)
            if len(x.shape) == 5:
                # [bs, T, ch, h, w]
                x = x.view(-1, *x.shape[2:])
            # forward pass
            model_output = model.training_step(x)
            # calculate loss
            loss = model_output['loss']
            rec_x = model_output['xrec']
            logs = model_output['logs_dict']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_kl = logs['kl_loss']
            loss_rec = logs['rec_loss']
            # log

            batch_losses.append(loss.data.cpu().item())
            batch_losses_rec.append(loss_rec.data.cpu().item())
            batch_losses_kl.append(loss_kl.data.cpu().item())
            # progress bar
            pbar.set_description_str(f'epoch #{epoch}')
            pbar.set_postfix(loss=loss.data.cpu().item(), rec=loss_rec.data.cpu().item(),
                             kl=loss_kl.data.cpu().item())
            # break  # for debug
        pbar.close()
        losses.append(np.mean(batch_losses))
        losses_rec.append(np.mean(batch_losses_rec))
        losses_kl.append(np.mean(batch_losses_kl))
        # scheduler
        scheduler.step()

        # epoch summary
        log_str = f'epoch {epoch} summary\n'
        log_str += f'loss: {losses[-1]:.3f}, rec: {losses_rec[-1]:.3f}, kl: {losses_kl[-1]:.3f}\n'
        log_str += f'val loss (freq: {eval_epoch_freq}): {valid_loss:.3f},' \
                   f' best: {best_valid_loss:.3f} @ epoch: {best_valid_epoch}\n'
        if eval_im_metrics:
            log_str += f'val lpips (freq: {eval_epoch_freq}): {val_lpips:.3f},' \
                       f' best: {best_val_lpips:.3f} @ epoch: {best_val_lpips_epoch}\n'
        print(log_str)
        log_line(log_dir, log_str)

        if epoch % eval_epoch_freq == 0 or epoch == num_epochs - 1:
            max_imgs = 8
            rec_x = model.to_rgb(rec_x)
            vutils.save_image(torch.cat([x[:max_imgs, -3:], rec_x[:max_imgs, -3:]],
                                        dim=0).data.cpu(), '{}/image_{}.jpg'.format(fig_dir, epoch),
                              nrow=8, pad_value=1)
            torch.save(model.state_dict(), os.path.join(save_dir, f'{ds}_{model_type}{run_prefix}.pth'))
            print("validation step...")
            torch.cuda.empty_cache()
            valid_loss = evaluate_validation_loss(model, config, epoch, batch_size=batch_size,
                                                  device=device,
                                                  save_image=True, fig_dir=fig_dir,
                                                  beta_rec=beta_rec,
                                                  beta_kl=beta_kl)
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
                                        f'{ds}_{model_type}{run_prefix}_best.pth'))
            torch.cuda.empty_cache()
            if eval_im_metrics and epoch > 0:
                valid_imm_results = eval_im_metric(model, device, config,
                                                   val_mode='val',
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
                               os.path.join(save_dir, f'{ds}_{model_type}{run_prefix}_best_lpips.pth'))
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

    conf_path = './vae/configs/panda.json'
    train_vae(conf_path)
