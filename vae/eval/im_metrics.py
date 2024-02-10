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