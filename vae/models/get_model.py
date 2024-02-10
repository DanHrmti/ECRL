# imports
import torch
from vae.models.vqvae import VQModel
from vae.models.vae import VAEModel


def get_model(config):
    model_type = config['model_type']
    if model_type == 'vqvae':
        model = VQModel(embed_dim=config['embed_dim'],
                        n_embed=config['n_embed'],
                        double_z=False,
                        z_channels=config['z_channels'],
                        resolution=config['image_size'],
                        in_channels=config['ch'],
                        out_ch=config['ch'],
                        ch=config['base_ch'],
                        ch_mult=config['ch_mult'],  # num_down = len(ch_mult)-1
                        num_res_blocks=config['num_res_blocks'],
                        attn_resolutions=config['attn_resolutions'],
                        dropout=config['dropout'],
                        device=torch.device(config['device']),
                        ckpt_path=config['pretrained_path'],
                        ignore_keys=[],
                        remap=None,
                        sane_index_shape=False)

    elif model_type == 'vae':
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
    else:
        raise NotImplemented(f'model: {model_type} not implemented')

    return model
