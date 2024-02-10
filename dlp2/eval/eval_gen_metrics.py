import os
import argparse
import json
from tqdm import tqdm
from dlp2.models import ObjectDLP
# datasets
from dlp2.datasets.panda_ds import PandaPush

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from piqa import PSNR, LPIPS, SSIM
except ImportError:
    print("piqa library required to compute image metrics")
    raise SystemExit


class ImageMetrics(nn.Module):
    """
    A class to calculate visual metrics between generated and ground-truth media
    """

    def __init__(self, metrics=('ssim', 'psnr', 'lpips')):
        super().__init__()
        self.metrics = metrics
        self.ssim = SSIM(reduction='none') if 'ssim' in self.metrics else None
        self.psnr = PSNR(reduction='none') if 'psnr' in self.metrics else None
        self.lpips = LPIPS(network='vgg', reduction='none') if 'lpips' in self.metrics else None

    @torch.no_grad()
    def forward(self, x, y):
        # x, y: [batch_size, 3, im_size, im_size] in [0,1]
        results = {}
        if self.ssim is not None:
            results['ssim'] = self.ssim(x, y)
        if self.psnr is not None:
            results['psnr'] = self.psnr(x, y)
        if self.lpips is not None:
            results['lpips'] = self.lpips(x, y)
        return results


def eval_im_metric(model, device, data_root_dir, val_mode='val', ds='shapes', eval_dir='./',
                   metrics=('ssim', 'psnr', 'lpips'), batch_size=32):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.eval()
    # load data
    if ds == "panda":
        image_size = 128
        dataset = PandaPush(data_root_dir, mode='valid', res=image_size)
    else:
        raise NotImplementedError

    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=0, drop_last=False)

    # image metric instance
    evaluator = ImageMetrics(metrics=metrics).to(device)

    # print(len(dataloader))
    results = {}
    ssims = []
    psnrs = []
    lpipss = []
    for i, batch in enumerate(tqdm(dataloader)):
        if ds == 'shapes':
            x = batch[0].to(device)
            x_prior = x
            idx_batch = batch[1]
        elif ds == 'panda':
            x = batch[0].squeeze(1).to(device)
            x_prior = x
        else:
            raise SyntaxError(f'dataset: {ds} not recognized')
        with torch.no_grad():
            output = model(x, x_prior=x_prior, deterministic=True)
            generated = output['rec'].clamp(0, 1)
            assert x.shape[1] == generated.shape[1], "prediction and gt frames shape don't match"
            results = evaluator(x, generated)
        # [batch_size * T]
        if 'ssim' in metrics:
            ssims.append(results['ssim'])
        if 'psnr' in metrics:
            psnrs.append(results['psnr'])
        if 'lpips' in metrics:
            lpipss.append(results['lpips'])

    if 'ssim' in metrics:
        mean_ssim = torch.cat(ssims, dim=0).mean().data.cpu().item()
        results['ssim'] = mean_ssim
    if 'psnr' in metrics:
        mean_psnr = torch.cat(psnrs, dim=0).mean().data.cpu().item()
        results['psnr'] = mean_psnr
    if 'lpips' in metrics:
        mean_lpips = torch.cat(lpipss, dim=0).mean().data.cpu().item()
        results['lpips'] = mean_lpips

    # save results
    path_to_conf = os.path.join(eval_dir, 'last_val_image_metrics.json')
    with open(path_to_conf, "w") as outfile:
        json.dump(results, outfile, indent=2)

    del evaluator  # clear memory

    return results


if __name__ == '__main__':
    # x = torch.rand(5, 3, 128, 128)
    # y = torch.rand(5, 3, 128, 128)
    # metrics = ImageMetrics()
    # results = metrics(x, y)
    # for k in results.keys():
    #     print(f'{k}: {results[k].shape}')
    parser = argparse.ArgumentParser(description="DDLP Video Prediction Balls Evaluation")
    parser.add_argument("-d", "--dataset", type=str, default='balls',
                        help="dataset to use: ['traffic', 'clevrer', 'shapes']")
    parser.add_argument("-p", "--path", type=str,
                        help="path to model directory, e.g. ./310822_141959_balls_dlp_dyn")
    parser.add_argument("--use_last", action='store_true',
                        help="use the last checkpoint instead of best")
    parser.add_argument("--use_train", action='store_true',
                        help="use the train set for the predictions")
    parser.add_argument("--cpu", action='store_true',
                        help="use cpu for inference")
    parser.add_argument("--prefix", type=str, default='',
                        help="prefix used for model saving")
    args = parser.parse_args()
    # parse input
    dir_path = args.path
    ds = args.dataset
    use_train = args.use_train
    # generation_horizon = args.horizon
    # num_predictions = args.num_predictions
    use_cpu = args.cpu
    deterministic = True
    prefix = args.prefix
    # load model config
    model_ckpt_name = f'{ds}_dlp{prefix}.pth'
    model_best_ckpt_name = f'{ds}_dlp{prefix}_best_lpips.pth'
    use_last = args.use_last if os.path.exists(os.path.join(dir_path, f'saves/{model_best_ckpt_name}')) else True
    conf_path = os.path.join(dir_path, 'hparams.json')
    with open(conf_path, 'r') as f:
        config = json.load(f)
    if use_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    # data properties
    if ds == "shapes" or ds == 'panda':
        image_size = 64
        ch = 3
        enc_channels = [32, 64, 128]
        prior_channels = (16, 32, 64)
    else:
        raise NotImplementedError
    # load model
    model = ObjectDLP(cdim=ch, enc_channels=enc_channels, prior_channels=prior_channels,
                      image_size=image_size, n_kp=config['n_kp'],
                      learned_feature_dim=config['learned_feature_dim'],
                      pad_mode=config['pad_mode'],
                      sigma=config['sigma'],
                      dropout=False, patch_size=config['patch_size'], n_kp_enc=config['n_kp_enc'],
                      n_kp_prior=config['n_kp_prior'], kp_range=config['kp_range'],
                      kp_activation=config['kp_activation'],
                      anchor_s=config['anchor_s'],
                      use_resblock=False,
                      scale_std=config['scale_std'],
                      offset_std=config['offset_std'], obj_on_alpha=config['obj_on_alpha'],
                      obj_on_beta=config['obj_on_beta']).to(device)
    ckpt_path = os.path.join(dir_path, f'saves/{model_ckpt_name if use_last else model_best_ckpt_name}')
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"loaded model from {ckpt_path}")

    # create dir for results
    pred_dir = os.path.join(dir_path, 'eval')
    os.makedirs(pred_dir, exist_ok=True)

    # conditional frames

    results = eval_im_metric(model, device, val_mode='test', ds=ds, eval_dir=pred_dir,
                             metrics=('ssim', 'psnr', 'lpips'), batch_size=10)
    print(f'results: {results}')
