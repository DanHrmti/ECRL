import os
import json
import pickle
from argparse import Namespace

import numpy as np
import matplotlib.pyplot as plt
import torch

from stable_baselines3.common.running_mean_std import RunningMeanStd

from dlp2.models import ObjectDLP
from dlp2.utils.util_func import plot_keypoints_on_image
from vae.models.vae import VAEModel
from slot_attention.method import SlotAttentionMethod
from slot_attention.slot_attention_model import SlotAttentionModel
from latent_classifier import MLPClassifier

from slot_attention.utils import to_rgb_from_tensor, sa_segment
from torchvision import utils as vutils


"""
Misc
"""


def check_config(config, isaac_env_cfg=None, policy_config=None):
    method = config['Model']['method']
    obs_type = config['Model']['obsType']
    obs_mode = config['Model']['obsMode']

    assert method in ['ECRL', 'SMORL', 'Unstructured']
    assert obs_type in ['State', 'Image']

    if obs_type == 'State':
        if method in ['ECRL', 'SMORL']:
            assert obs_mode == 'state'
        if method == 'Unstructured':
            assert obs_mode == 'state_unstruct'

    if obs_type == 'Image':
        if method == 'ECRL':
            assert obs_mode in ['dlp', 'slot']
        if method == 'SMORL':
            assert obs_mode == 'dlp' and config['Model']['ChamferReward']
        if method == 'Unstructured':
            assert obs_mode == 'vae'

    if config['Model']['ChamferReward']:
        assert method in ['ECRL', 'SMORL'] and obs_type == 'Image' and obs_mode == 'dlp'

    # policy related
    if policy_config is not None:
        if obs_mode in ['state', 'slot']:
            assert not policy_config[method][obs_type]['actor_kwargs'].get('masking', False)

    # environment related
    if isaac_env_cfg is not None:
        if isaac_env_cfg["env"].get("PushT", False):
            assert isaac_env_cfg["env"].get("numColors", isaac_env_cfg["env"]["numObjects"]) == 1


def get_run_name(config, isaac_env_cfg, seed):
    name = f"{isaac_env_cfg['env']['numObjects']}C_{config['Model']['method']}_{config['Model']['obsType']}"
    if config['Model']['obsMode'] == 'slot':
        name += "_Slot"
    if config['Model']['ChamferReward']:
        name += "_ChamferReward"
    if isaac_env_cfg['env']['tableDims'][0] < 0.5:
        name += "_SmallTable"
    for key in ["AdjacentGoals", "OrderedPush", "PushT", "RandColor", "RandNumObj"]:
        if key in isaac_env_cfg['env'] and isaac_env_cfg['env'][key]:
            name += f"_{key}"
    return name


"""
Logging
"""


def compute_gradients(parameters):
    total_gradient_norm = None
    for p in parameters:
        # if p.grad is None:
        #     continue
        current = p.grad.data.norm(2) ** 2
        if total_gradient_norm is None:
            total_gradient_norm = current
        else:
            total_gradient_norm += current
    return total_gradient_norm ** 0.5


def compute_params(parameters):
    total_param_norm = None
    for p in parameters:
        current = p.data.norm(2) ** 2
        if total_param_norm is None:
            total_param_norm = current
        else:
            total_param_norm += current
    return total_param_norm ** 0.5


def get_max_param(parameters):
    max_p = 0
    for p in parameters:
        current = p.data.abs().max()
        if current > max_p:
            max_p = current
    return max_p


"""
Pretrained Representation
"""


def load_pretrained_rep_model(dir_path, model_type='dlp'):

    if model_type not in ['dlp', 'vae', 'slot']:
        return None

    ckpt_path = os.path.join(dir_path, f'{model_type}_panda_push.pth')

    if model_type == 'dlp':
        print("\nLoading pretrained DLP...")
        # load config
        conf_path = os.path.join(dir_path, 'hparams.json')
        with open(conf_path, 'r') as f:
            config = json.load(f)
        # initialize model
        model = ObjectDLP(cdim=config['cdim'], enc_channels=config['enc_channels'],
                          prior_channels=config['prior_channels'],
                          image_size=config['image_size'], n_kp=config['n_kp'],
                          learned_feature_dim=config['learned_feature_dim'],
                          bg_learned_feature_dim=config['bg_learned_feature_dim'],
                          pad_mode=config['pad_mode'],
                          sigma=config['sigma'],
                          dropout=False, patch_size=config['patch_size'], n_kp_enc=config['n_kp_enc'],
                          n_kp_prior=config['n_kp_prior'], kp_range=config['kp_range'],
                          kp_activation=config['kp_activation'],
                          anchor_s=config['anchor_s'],
                          use_resblock=False,
                          scale_std=config['scale_std'],
                          offset_std=config['offset_std'], obj_on_alpha=config['obj_on_alpha'],
                          obj_on_beta=config['obj_on_beta'])
        # load model from checkpoint
        model.load_state_dict(torch.load(ckpt_path))

    elif model_type == 'vae':
        print("\nLoading pretrained VAE...")
        # load config
        conf_path = os.path.join(dir_path, 'hparams.json')
        with open(conf_path, 'r') as f:
            config = json.load(f)
        # initialize model
        model = VAEModel(double_z=False,
                         z_channels=config['z_channels'],
                         resolution=config['image_size'],
                         in_channels=config['ch'],
                         out_ch=config['ch'],
                         ch=config['base_ch'],
                         ch_mult=config['ch_mult'],  # num_down = len(ch_mult)-1
                         num_res_blocks=config['num_res_blocks'],
                         attn_resolutions=config['attn_resolutions'],
                         dropout=config['dropout'],
                         latent_dim=config['latent_dim'],
                         kl_weight=config['beta_kl'],
                         device=torch.device(config['device']),
                         ckpt_path=config['pretrained_path'],
                         ignore_keys=[],
                         remap=None,
                         sane_index_shape=False)
        # load model from checkpoint
        model.load_state_dict(torch.load(ckpt_path))
        del model.loss
        model.loss = None

    elif model_type == 'slot':
        print("\nLoading pretrained Slot-Attention...")
        # load config
        ckpt = torch.load(ckpt_path)
        params = Namespace(**ckpt["hyper_parameters"])
        # initialize model
        sa = SlotAttentionModel(
            resolution=params.resolution,
            num_slots=params.num_slots,
            num_iterations=params.num_iterations,
            slot_size=params.slot_size,
        )
        # load model from checkpoint
        model = SlotAttentionMethod.load_from_checkpoint(ckpt_path, model=sa, datamodule=None)

    else:
        raise NotImplementedError(f"Pretrained model type '{model_type}' is not supported")

    model.eval()
    model.requires_grad_(False)

    print(f"Loaded pretrained representation model from {ckpt_path}\n")

    return model


def get_dlp_rep(dlp_output):
    pixel_xy = dlp_output['z']
    scale_xy = dlp_output['mu_scale']
    depth = dlp_output['mu_depth']
    visual_features = dlp_output['mu_features']
    transp = dlp_output['obj_on'].unsqueeze(dim=-1)
    rep = torch.cat((pixel_xy, scale_xy, depth, visual_features, transp), dim=-1)
    return rep


def extract_dlp_image(images, latent_rep_model, device):
    orig_image_shape = images.shape
    if len(orig_image_shape) == 3:
        images = np.expand_dims(images, axis=0)
    normalized_images = images.astype('float32') / 255
    normalized_images = torch.tensor(normalized_images, device=device)

    with torch.no_grad():
        encoded_output = latent_rep_model.encode_all(normalized_images, deterministic=True)
        pixel_xy = encoded_output['z']

    dlp_images = []
    for kp_xy, image in zip(pixel_xy, normalized_images):
        dlp_images.append(
            plot_keypoints_on_image(kp_xy, image, radius=2, thickness=1, kp_range=(-1, 1), plot_numbers=False))

    if len(dlp_images) == 1:
        dlp_images = dlp_images[0]

    return dlp_images


def extract_slot_image(images, latent_rep_model, device):
    normalized_images = images.astype('float32') / 255
    normalized_images = torch.tensor(normalized_images, device=device)
    normalized_images = normalized_images * 2 - 1
    recon_combined, recons, masks, slots = latent_rep_model.forward(normalized_images)
    # `masks` has shape [batch_size, num_entries, channels, height, width].
    threshold = getattr(latent_rep_model.params, "sa_segmentation_threshold", 0.5)
    _, _, cmap_segmentation, cmap_segmentation_thresholded = sa_segment(
        masks, threshold
    )

    # combine images in a nice way so we can display all outputs in one grid, output rescaled to be between 0 and 1
    out = torch.cat(
        [
            to_rgb_from_tensor(normalized_images.unsqueeze(1)),  # original images
            to_rgb_from_tensor(recon_combined.unsqueeze(1)),  # reconstructions
            cmap_segmentation.unsqueeze(1),
            cmap_segmentation_thresholded.unsqueeze(1),
            to_rgb_from_tensor(recons * masks + (1 - masks)),  # each slot
        ],
        dim=1,
    )

    batch_size, num_slots, C, H, W = recons.shape
    images = vutils.make_grid(
        out.view(batch_size * out.shape[1], C, H, W).cpu(),
        normalize=False,
        nrow=out.shape[1],
    )

    return images


"""
Reward
"""


def batch_pairwise_dist(x, y, metric='l2_simple'):
    assert metric in ['l2', 'l2_simple', 'l1', 'cosine'], f'metric {metric} unrecognized'
    bs, num_points_x, points_dim = x.size()
    _, num_points_y, _ = y.size()
    if metric == 'cosine':
        dist_func = torch.nn.functional.cosine_similarity
        P = -dist_func(x.unsqueeze(2), y.unsqueeze(1), dim=-1, eps=1e-8)
    elif metric == 'l1':
        P = torch.abs(x.unsqueeze(2) - y.unsqueeze(1)).sum(-1)
    elif metric == 'l2_simple':
        P = ((x.unsqueeze(2) - y.unsqueeze(1)) ** 2).sum(-1)
    else:
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        diag_ind_x = torch.arange(0, num_points_x, device=x.device)
        diag_ind_y = torch.arange(0, num_points_y, device=y.device)
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = rx.transpose(2, 1) + ry - 2 * zz
    return P


def get_bb_from_z_scale(kp, scale):
    # extracts bounding boxes (bb) from keypoints and scales.
    # kp: [n_kp, 2], range: (-1, 1)
    # z_scale: [n_kp, 2], range: (0, 1)
    n_kp = kp.shape[0]
    coor = torch.zeros(size=(n_kp, 4), dtype=torch.float32, device=kp.device)
    # normalize
    kp_norm = 0.5 + kp / 2  # [0, 1]
    scale_norm = torch.sigmoid(scale)

    x_kp = kp_norm[:, 0]
    x_scale = scale_norm[:, 0]
    y_kp = kp_norm[:, 1]
    y_scale = scale_norm[:, 1]

    ws = (x_kp - x_scale / 2).clamp(0, 1)
    wt = (x_kp + x_scale / 2).clamp(0, 1)
    hs = (y_kp - y_scale / 2).clamp(0, 1)
    ht = (y_kp + y_scale / 2).clamp(0, 1)

    coor[:, 0] = ws
    coor[:, 1] = hs
    coor[:, 2] = wt
    coor[:, 3] = ht
    return coor


def load_latent_classifier(config, num_objects):
    if config['Model']['obsMode'] == 'dlp' and (config['Model']['ChamferReward'] or config['Model']['method'] == 'SMORL'):
        dir_path = config['Reward']['LatentClassifier']['path']
        latent_classifier_chkpt_path = f'{dir_path}/latent_classifier_{num_objects}C_dlp_push_5C'
        latent_classifier = MLPClassifier(**config['Reward']['LatentClassifier']['params'])
        latent_classifier.mlp.load_state_dict(torch.load(latent_classifier_chkpt_path))
        print(f"Loaded latent_classifier model from {latent_classifier_chkpt_path}")
        return latent_classifier


"""
Agent
"""


def action_noise_schedule(sig_start, sig_end, init_episodes, ss_episodes, tot_episodes):
    noise_schedule = []
    if init_episodes > 0:
        init_sigmas = np.ones(init_episodes) * sig_start
        noise_schedule.extend(init_sigmas)
    linear_sch_sigmas = np.linspace(sig_start, sig_end, tot_episodes - init_episodes - ss_episodes)
    noise_schedule.extend(linear_sch_sigmas)
    if ss_episodes > 0:
        ss_sigmas = np.ones(ss_episodes) * sig_end
        noise_schedule.extend(ss_sigmas)
    return np.asarray(noise_schedule)


class RMSNormalizer:
    def __init__(self, epsilon=1e-6, shape=()):
        self.epsilon = epsilon
        self.rms = RunningMeanStd(epsilon=epsilon, shape=shape)

    def update(self, obs):
        self.rms.update(obs)

    def normalize(self, obs):
        if torch.is_tensor(obs):
            device = obs.device
            dtype = obs.dtype
            mean = torch.tensor(self.rms.mean, device=device, dtype=dtype)
            var = torch.tensor(self.rms.var, device=device, dtype=dtype)
            epsilon = torch.tensor(self.epsilon, device=device, dtype=dtype)
            return torch.clip((obs - mean) / torch.sqrt(var + epsilon), -5, 5).to(torch.float32)
        else:
            return np.clip((obs - self.rms.mean) / np.sqrt(self.rms.var + self.epsilon), -5, 5).astype(np.float32)

    def unnormalize(self, obs):
        if torch.is_tensor(obs):
            device = obs.device
            dtype = obs.dtype
            mean = torch.tensor(self.rms.mean, device=device, dtype=dtype)
            var = torch.tensor(self.rms.var, device=device, dtype=dtype)
            epsilon = torch.tensor(self.epsilon, device=device, dtype=dtype)
            return (obs * torch.sqrt(var + epsilon)) + mean
        else:
            return (obs * np.sqrt(self.rms.var + self.epsilon)) + self.rms.mean
