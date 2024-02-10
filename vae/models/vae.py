"""
VQ-VAE-2
based on: https://github.com/CompVis/taming-transformers/blob/master/taming
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
# from einops import rearrange
from torchvision import models
from collections import namedtuple
import os
import hashlib
import requests
from PIL import Image
from tqdm import tqdm

URL_MAP = {
    "vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"
}

CKPT_MAP = {
    "vgg_lpips": "vgg.pth"
}

MD5_MAP = {
    "vgg_lpips": "d507d7349b931f0638a25a48a722f98a"
}

"""
Functions
"""


def calc_model_size(model):
    num_trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    # estimate model size on disk: https://discuss.pytorch.org/t/finding-model-size/130275/2
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    return {'n_params': num_trainable_params, 'size_mb': size_all_mb}


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


def md5_hash(path):
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def get_ckpt_path(name, root, check=False):
    assert name in URL_MAP
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
        print("Downloading {} model from {} to {}".format(name, URL_MAP[name], path))
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], md5
    return path


def normalize_tensor(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def spatial_average(x, keepdim=True):
    return x.mean([2, 3], keepdim=keepdim)


def norm_layer(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


def preprocess_vqvae(x):
    x = 2. * x - 1.
    return x


def v_to_rgb(x):
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    return x


def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


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


def calc_kl(logvar, mu, mu_o=0.0, logvar_o=0.0, reduce='mean', balance=0.5):
    """
    Calculate kl-divergence
    :param logvar: log-variance from the encoder
    :param mu: mean from the encoder
    :param mu_o: negative mean for outliers (hyper-parameter)
    :param logvar_o: negative log-variance for outliers (hyper-parameter)
    :param reduce: type of reduce: 'sum', 'none'
    :param balance: balancing coefficient between posterior and prior
    :return: kld
    """
    if not isinstance(mu_o, torch.Tensor):
        mu_o = torch.tensor(mu_o).to(mu.device)
    if not isinstance(logvar_o, torch.Tensor):
        logvar_o = torch.tensor(logvar_o).to(mu.device)
    if balance == 0.5:
        kl = -0.5 * (1 + logvar - logvar_o - logvar.exp() / torch.exp(logvar_o) - (mu - mu_o).pow(2) / torch.exp(
            logvar_o)).sum(1)
    else:
        # detach post
        mu_post = mu.detach()
        logvar_post = logvar.detach()
        mu_prior = mu_o
        logvar_prior = logvar_o
        kl_a = -0.5 * (1 + logvar_post - logvar_prior - logvar_post.exp() / torch.exp(logvar_prior) - (
                mu_post - mu_prior).pow(2) / torch.exp(logvar_prior)).sum(1)
        # detach prior
        mu_post = mu
        logvar_post = logvar
        mu_prior = mu_o.detach()
        logvar_prior = logvar_o.detach()
        kl_b = -0.5 * (1 + logvar_post - logvar_prior - logvar_post.exp() / torch.exp(logvar_prior) - (
                mu_post - mu_prior).pow(2) / torch.exp(logvar_prior)).sum(1)
        kl = (1 - balance) * kl_a + balance * kl_b
    if reduce == 'sum':
        kl = torch.sum(kl)
    elif reduce == 'mean':
        kl = torch.mean(kl)
    return kl


"""
Modules
"""


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = norm_layer(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = norm_layer(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = norm_layer(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = norm_layer(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2 * z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        # assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)

        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = norm_layer(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class LPIPS(nn.Module):
    # Learned perceptual metric
    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vg16 features
        self.net = vgg16(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name="vgg_lpips"):
        ckpt = get_ckpt_path(name, "eval/lpips")
        self.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
        print("loaded pretrained LPIPS loss from {}".format(ckpt))

    @classmethod
    def from_pretrained(cls, name="vgg_lpips"):
        if name != "vgg_lpips":
            raise NotImplementedError
        model = cls()
        ckpt = get_ckpt_path(name)
        model.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
        return model

    def forward(self, input, target):
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


class KLLPIPS(nn.Module):
    def __init__(self, kl_weight=1.0, pixelloss_weight=1.0, perceptual_weight=1.0):
        super().__init__()
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight

    def forward(self, kl_loss, inputs, reconstructions, split="train"):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor([0.0])

        nll_loss = rec_loss
        # nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        nll_loss = torch.mean(nll_loss)

        loss = nll_loss + self.kl_weight * kl_loss.mean()

        # log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
        #        "{}/quant_loss".format(split): codebook_loss.detach().mean(),
        #        "{}/nll_loss".format(split): nll_loss.detach().mean(),
        #        "{}/rec_loss".format(split): rec_loss.detach().mean(),
        #        "{}/p_loss".format(split): p_loss.detach().mean(),
        #        }
        log = {"total_loss".format(split): loss.clone().detach().mean(),
               "quant_loss".format(split): kl_loss.detach().mean(),
               "kl_loss".format(split): kl_loss.detach().mean(),
               "nll_loss".format(split): nll_loss.detach().mean(),
               "rec_loss".format(split): rec_loss.detach().mean(),
               "p_loss".format(split): p_loss.detach().mean(),
               }
        return loss, log


class VAEModel(nn.Module):
    def __init__(self,
                 double_z=False,
                 z_channels=256,
                 resolution=128,
                 in_channels=3,
                 out_ch=3,
                 ch=128,
                 ch_mult=(1, 1, 2, 2, 4),  # num_down = len(ch_mult)-1
                 num_res_blocks=2,
                 attn_resolutions=(16,),
                 dropout=0.0,
                 latent_dim=256,
                 kl_weight=1.0,
                 device=torch.device('cpu'),
                 ckpt_path=None,
                 ignore_keys=[],
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        # self.image_key = image_key
        self.in_ch = in_channels
        self.resolution = resolution
        self.device = device
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.encoder = Encoder(ch=ch, out_ch=out_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                               attn_resolutions=attn_resolutions, dropout=dropout, in_channels=in_channels,
                               resolution=resolution, z_channels=z_channels, double_z=double_z)
        self.decoder = Decoder(ch=ch, out_ch=out_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                               attn_resolutions=attn_resolutions, dropout=dropout, in_channels=in_channels,
                               resolution=resolution, z_channels=z_channels, double_z=double_z)
        self.loss = KLLPIPS(kl_weight=self.kl_weight, pixelloss_weight=1.0, perceptual_weight=1.0)
        self.enc_output_shape = self.get_conv_shape()
        self.enc_output_shape_flat = np.prod(self.enc_output_shape)
        self.to_latent = nn.Sequential(nn.Linear(self.enc_output_shape_flat, 512),
                                       nn.ReLU(True), nn.Linear(512, 2 * self.latent_dim))
        self.from_latent = nn.Sequential(nn.Linear(self.latent_dim, 512),
                                         nn.ReLU(True),
                                         nn.Linear(512, self.enc_output_shape_flat))
        # self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
        #                                 remap=remap, sane_index_shape=sane_index_shape)
        # self.quant_conv = torch.nn.Conv2d(z_channels, embed_dim, 1)

        # self.post_quant_conv = torch.nn.Conv2d(embed_dim, z_channels, 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        # self.image_key = image_key
        # if colorize_nlabels is not None:
        #     assert type(colorize_nlabels) == int
        #     self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        # if monitor is not None:
        #     self.monitor = monitor

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x, deterministic=False):
        h = self.encoder(x)
        h = h.view(h.shape[0], -1)
        h = self.to_latent(h)
        mu, logvar = h.chunk(2, dim=-1)
        if deterministic:
            z = mu
        else:
            z = reparameterize(mu, logvar)
        return z, mu, logvar

    def get_latent_rep(self, x, deterministic=True):
        z, _, _ = self.encode(x, deterministic=deterministic)
        return z

    def decode(self, z):
        z = self.from_latent(z)
        z = z.view(z.shape[0], *self.enc_output_shape)
        dec = self.decoder(z)
        return dec

    # def decode_code(self, code_b):
    #     quant_b = self.quantize.embed_code(code_b)
    #     dec = self.decode(quant_b)
    #     return dec

    def forward(self, x, deterministic=False):
        # x: [bs, ch, h, w] in [0, 1]
        z, mu, logvar = self.encode(x, deterministic)
        kl_loss = calc_kl(logvar, mu, reduce='mean')
        dec = self.decode(z)
        return dec, kl_loss

    # def get_input(self, batch, k):
    #     x = batch[k]
    #     if len(x.shape) == 3:
    #         x = x[..., None]
    #     x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
    #     return x.float()

    def training_step(self, x, deterministic=False):
        # x = self.get_input(batch, self.image_key)
        x = self.preprocess_rgb(x)  # -> [-1, 1]
        xrec, qloss = self(x, deterministic)

        # autoencode
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="train")
        # log_str = f'train/aeloss: {aeloss:.4f}'
        # self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        # self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        out = {'loss': aeloss,
               'logs_dict': log_dict_ae,
               'xrec': xrec}
        return out

    def validation_step(self, x, deterministic=True):
        # x = self.get_input(batch, self.image_key)
        x = self.preprocess_rgb(x)  # -> [-1, 1]
        xrec, qloss = self(x, deterministic)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="val")
        # rec_loss = log_dict_ae["val/rec_loss"]
        # log_str = f'val/rec_loss: {rec_loss:.4f}, val/aeloss: {aeloss:.4f}'
        # self.log("val/rec_loss", rec_loss,
        #          prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        # self.log("val/aeloss", aeloss,
        #          prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        # self.log_dict(log_dict_ae)
        out = {'loss': aeloss,
               'logs_dict': log_dict_ae,
               'xrec': xrec}
        return out

    # def configure_optimizers(self):
    #     lr = self.learning_rate
    #     opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
    #                               list(self.decoder.parameters()) +
    #                               list(self.quantize.parameters()) +
    #                               list(self.quant_conv.parameters()) +
    #                               list(self.post_quant_conv.parameters()),
    #                               lr=lr, betas=(0.5, 0.9))
    #     opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
    #                                 lr=lr, betas=(0.5, 0.9))
    #     return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, x):
        log = dict()
        # x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _ = self(x)
        # if x.shape[1] > 3:
        #     # colorize with random projection
        #     assert xrec.shape[1] > 3
        #     x = self.to_rgb(x)
        #     xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def preprocess_rgb(self, x):
        return preprocess_vqvae(x)  # -> [-1, 1]

    def to_rgb(self, x):
        return v_to_rgb(x)

    def get_conv_shape(self):
        dummy_x = torch.rand(1, self.in_ch, self.resolution, self.resolution)
        with torch.no_grad():
            z = self.encoder(dummy_x)
        return z.shape[1:]

    def info(self):
        dummy_x = torch.rand(1, self.in_ch, self.resolution, self.resolution, device=self.device)
        with torch.no_grad():
            z, _, _ = self.encode(dummy_x)
        log_str = f'latent shape: {z.shape}\n'
        # encoder size
        enc_size_dict = calc_model_size(self.encoder)
        enc_n_params = enc_size_dict['n_params']
        log_str += f'encoder trainable parameters: {enc_n_params} ({enc_n_params / (10 ** 6):.4f}M)\n'
        # decoder size
        dec_size_dict = calc_model_size(self.decoder)
        dec_n_params = dec_size_dict['n_params']
        log_str += f'decoder trainable parameters: {dec_n_params} ({dec_n_params / (10 ** 6):.4f}M)\n'
        # dynamic module
        # num parameters and model size
        size_dict = calc_model_size(self)
        size_mb = size_dict['size_mb']
        n_params = size_dict['n_params']
        log_str += f'total model trainable parameters: {n_params} ({n_params / (10 ** 6):.4f}M)\n'
        log_str += f'total model estimated size on disk: {size_mb:.3f}MB\n'
        return log_str

    # def to_rgb(self, x):
    #     assert self.image_key == "segmentation"
    #     if not hasattr(self, "colorize"):
    #         self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
    #     x = F.conv2d(x, weight=self.colorize)
    #     x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
    #     return x


if __name__ == '__main__':
    n_embed = 1024
    double_z = False
    embed_dim = 16
    z_channels = 64
    resolution = 128
    in_channels = 3
    out_ch = 3
    ch = 64
    ch_mult = (1, 1, 2, 2, 2, 4)  # num_down = len(ch_mult)-1
    num_res_blocks = 2
    attn_resolutions = (16,)
    dropout = 0.0
    latent_dim = 256
    kl_weight = 0.001
    device = torch.device('cpu')
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = VAEModel(double_z,
                    z_channels,
                    resolution,
                    in_channels,
                    out_ch,
                    ch,
                    ch_mult,
                    num_res_blocks,
                    attn_resolutions,
                    dropout,
                    latent_dim,
                    kl_weight,
                    device).to(device)
    print(model.info())

    dummy_in = torch.rand(1, in_channels, resolution, resolution, device=device)
    dec, diff = model(dummy_in)
    print(f'dec: {dec.shape} in [{dec.min():.3f}, {dec.max():.3f}], kl: {diff}')
