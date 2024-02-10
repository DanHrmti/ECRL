import os
import pickle
from tqdm.auto import tqdm
import yaml
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from latent_classifier import MLPClassifier

from utils import RMSNormalizer, load_pretrained_rep_model, batch_pairwise_dist, get_bb_from_z_scale
from dlp2.utils.util_func import plot_keypoints_on_image


"""
Helpers
"""


def get_dlp_rep(dlp_output):
    pixel_xy = dlp_output['z']
    scale_xy = dlp_output['mu_scale']
    depth = dlp_output['mu_depth']
    visual_features = dlp_output['mu_features']
    transp = dlp_output['obj_on'].unsqueeze(dim=-1)
    return torch.cat((pixel_xy, scale_xy, depth, visual_features, transp), dim=-1)


def extract_dlp_features(obs, dlp_model):
    normalized_observations = obs.to(torch.float32) / 255

    with torch.no_grad():
        encoded_output = dlp_model.encode_all(normalized_observations, deterministic=True)
        particles = get_dlp_rep(encoded_output)

    return particles


def convert_obs_to_dlp_rep(obs, dlp_model):
    num_viewpoints = obs.shape[1]
    dlp_obs = [extract_dlp_features(obs[:, i], dlp_model) for i in range(num_viewpoints)]
    particles = torch.cat([torch.unsqueeze(view, axis=1) for view in dlp_obs], axis=1)
    return particles


def extract_dlp_image(images, dlp_model, reward_model):
    normalized_images = images.to(torch.float32) / 255

    with torch.no_grad():
        encoded_output = dlp_model.encode_all(normalized_images, deterministic=True)
        dlp_features = get_dlp_rep(encoded_output)
        dlp_features_normalized = reward_model.particle_normalizer.normalize(dlp_features)
        normalized_obj_on = dlp_features_normalized[..., -1]
        pixel_xy = dlp_features[..., :2]
        mask = reward_model.get_particle_mask(dlp_features, normalized_obj_on)
        pixel_xy = [pixel_xy[i][~mask[i]] for i in range(len(pixel_xy))]

        dlp_images = []
        for kp_xy, image in zip(pixel_xy, normalized_images):
            dlp_images.append(plot_keypoints_on_image(kp_xy, image, radius=2, thickness=1, kp_range=(-1, 1), plot_numbers=False))

    return dlp_images


def plot_reward(reward, images, goal_images, dlp_model, reward_model=None):
    bs, n_views, c, h, w = images.shape

    dlp_image_views = [extract_dlp_image(images[:, i], dlp_model, reward_model) for i in range(n_views)]
    dlp_goal_image_views = [extract_dlp_image(goal_images[:, i], dlp_model, reward_model) for i in range(n_views)]

    n_row, n_col = 2 * n_views, bs

    fig = plt.figure(figsize=(7 * n_col, 7 * n_row))
    fig.suptitle(f"Reward Visualization", y=0.95, fontsize=30)

    for j in range(n_col):
        for i in range(n_row):
            ax = fig.add_subplot(n_row, n_col, i * n_col + (j + 1))
            view_idx = i // 2
            state_goal_idx = i % 2
            if state_goal_idx == 0:
                ax.imshow(dlp_image_views[view_idx][j])
                if j == 0:
                    ax.set_title(f"State - View {view_idx}", y=-0.1, loc='left', fontsize=20)
            else:
                ax.imshow(dlp_goal_image_views[view_idx][j])
                if j == 0:
                    ax.set_title(f"Goal - View {view_idx}", y=-0.1, loc='left', fontsize=20)

            if i == 0:
                ax.set_title(f"Reward = {reward[j]:1.4f}", fontsize=30)

            ax.set_axis_off()

    plt.show()
    return


"""
Reward Models
"""


class ChamferReward(nn.Module):
    def __init__(self, particle_normalizer, latent_classifier=None, reward_scale=1.0, dist_norm=2,
                 chamfer_metric='l2_simple', latent_dist_threshold=6, smorl=False):
        super(ChamferReward, self).__init__()
        self.particle_normalizer = particle_normalizer
        self.reward_scale = reward_scale
        self.smorl = smorl
        self.chamfer_metric = chamfer_metric
        self.latent_dist_threshold = latent_dist_threshold
        self.dist_norm = dist_norm

        self.latent_classifier = latent_classifier

    def forward(self, obs, unnormalize=True):
        state_particles = obs["achieved_goal"]
        goal_particles = obs["desired_goal"]

        bs, n_views, n_particles, feature_dim = state_particles.shape

        if unnormalize:
            # unnormalize particles
            state_particles = self.particle_normalizer.unnormalize(state_particles)
            goal_particles = self.particle_normalizer.unnormalize(goal_particles)

        # create masks
        state_mask = self.get_particle_mask(state_particles)
        goal_mask = self.get_particle_mask(goal_particles)

        if not self.smorl:
            P_state_mask = state_mask.unsqueeze(-1).expand(-1, -1, -1, n_particles)
            P_goal_mask = goal_mask.unsqueeze(-1).expand(-1, -1, -1, n_particles).transpose(-1, -2)

            P_mask = torch.logical_or(P_state_mask, P_goal_mask)

        else:
            P_mask = state_mask.unsqueeze(-1).to(torch.bool)

        # compute reward
        state_particle_vis_features = state_particles[..., 5:9]
        state_particles_xy = state_particles[..., :2]
        goal_particle_vis_features = goal_particles[..., 5:9]
        goal_particles_xy = goal_particles[..., :2]

        reward_view = state_particles.new_zeros([n_views, bs])
        for i in range(n_views):
            state_particles, goal_particles = state_particles_xy[:, i], goal_particles_xy[:, i]
            P = batch_pairwise_dist(state_particle_vis_features[:, i], goal_particle_vis_features[:, i], self.chamfer_metric)
            P.masked_fill_(P_mask[:, i], float('inf'))  # disregard particles that don't represent objects in min operation

            # compute dist from goal to state
            min_latent_dists, min_indices = torch.min(P, 1)
            latent_dist_mask_1 = min_latent_dists > self.latent_dist_threshold
            a_goal = goal_particles
            d_goal = torch.gather(state_particles, 1, min_indices.unsqueeze(-1).expand(-1, -1, 2))
            xy_dist_1 = torch.linalg.norm(a_goal - d_goal, ord=self.dist_norm, dim=-1)
            xy_dist_1[latent_dist_mask_1] = 1  # set max dist for particles with no match
            xy_dist_1[goal_mask[:, i]] = 0  # filter particles' contribution to reward based on goal mask
            n_object_particles = (torch.sum(~goal_mask[:, i], 1))
            n_object_particles = torch.maximum(n_object_particles, torch.ones_like(n_object_particles))  # make sure we don't divide by zero
            reward_g2s = - torch.sum(xy_dist_1, 1) / n_object_particles  # normalize based on number of unfiltered particles
            if self.smorl:
                reward_view[i] = reward_g2s
                continue  # only consider distance from single goal to state

            # compute dist from state to goal
            min_latent_dists, min_indices = torch.min(P, 2)
            latent_dist_mask_2 = min_latent_dists > self.latent_dist_threshold
            a_goal = state_particles
            d_goal = torch.gather(goal_particles, 1, min_indices.unsqueeze(-1).expand(-1, -1, 2))
            xy_dist_2 = torch.linalg.norm(a_goal - d_goal, ord=self.dist_norm, dim=-1)
            xy_dist_2[latent_dist_mask_2] = 1  # set max dist for particles with no match
            xy_dist_2[state_mask[:, i]] = 0  # filter particles' contribution to reward based on mask
            n_object_particles = (torch.sum(~state_mask[:, i], 1))
            n_object_particles = torch.maximum(n_object_particles, torch.ones_like(n_object_particles))  # make sure we don't divide by zero
            reward_s2g = - torch.sum(xy_dist_2, 1) / n_object_particles  # normalize based on number of unfiltered particles

            # take mean over chamfer rewards
            reward_view[i] = (reward_g2s + reward_s2g) / 2
            # set max negative reward in case there are no valid particles in either state or goal
            no_valid_particles = torch.logical_or(torch.all(goal_mask[:, i], dim=-1), torch.all(state_mask[:, i], dim=-1))
            reward_view[i][no_valid_particles] = -1

        reward = torch.mean(reward_view, dim=0)

        return self.reward_scale * reward.unsqueeze(-1)

    def get_particle_mask(self, particles):

        if len(particles.shape) == 4:
            bs, n_views, n_particles, feature_dim = particles.shape
        else:
            bs, n_particles, feature_dim = particles.shape
            n_views = 1

        pixel_xy = particles[..., :2]
        scale_xy = particles[..., 2:4]
        vis_features = particles[..., 5:9]
        obj_on = particles[..., -1]

        # latent classifier condition
        if self.latent_classifier is not None:
            obj_class_cond = self.latent_classifier.classify(vis_features)
        else:
            obj_class_cond = torch.ones_like(obj_on)

        mask = (obj_class_cond == 0)

        return mask


class DensityAwareChamferReward(nn.Module):
    def __init__(self, particle_normalizer, latent_classifier=None, reward_scale=1.0, dist_norm=2,
                 chamfer_metric='l2_simple', latent_dist_threshold=6):
        super(DensityAwareChamferReward, self).__init__()
        self.particle_normalizer = particle_normalizer
        self.reward_scale = reward_scale
        self.chamfer_metric = chamfer_metric
        self.latent_dist_threshold = latent_dist_threshold
        self.dist_norm = dist_norm

        self.latent_classifier = latent_classifier

    def forward(self, obs, unnormalize=True):
        state_particles = obs["achieved_goal"]
        goal_particles = obs["desired_goal"]

        bs, n_views, n_particles, feature_dim = state_particles.shape

        if unnormalize:
            # unnormalize particles
            state_particles = self.particle_normalizer.unnormalize(state_particles)
            goal_particles = self.particle_normalizer.unnormalize(goal_particles)

        # create masks
        state_mask = self.get_particle_mask(state_particles)
        goal_mask = self.get_particle_mask(goal_particles)

        P_state_mask = state_mask.unsqueeze(-1).expand(-1, -1, -1, n_particles)
        P_goal_mask = goal_mask.unsqueeze(-1).expand(-1, -1, -1, n_particles).transpose(-1, -2)

        P_mask = torch.logical_or(P_state_mask, P_goal_mask)

        # compute reward
        state_particle_vis_features = state_particles[..., 5:9]
        state_particles_xy = state_particles[..., :2]
        goal_particle_vis_features = goal_particles[..., 5:9]
        goal_particles_xy = goal_particles[..., :2]

        reward_view = state_particles.new_zeros([n_views, bs])
        for i in range(n_views):
            state_particles, goal_particles = state_particles_xy[:, i], goal_particles_xy[:, i]
            P = batch_pairwise_dist(state_particle_vis_features[:, i], goal_particle_vis_features[:, i], self.chamfer_metric)
            P.masked_fill_(P_mask[:, i], float('inf'))  # disregard particles that don't represent objects in min operation

            # Goal -> State distance
            # compute distance to closest matching particle
            min_dists, min_indices = torch.min(P, 1)
            latent_dist_mask_1 = min_dists > self.latent_dist_threshold
            a_goal = goal_particles
            d_goal = torch.gather(state_particles, 1, min_indices.unsqueeze(-1).expand(-1, -1, state_particles.shape[-1]))
            dist_1 = torch.linalg.norm(a_goal - d_goal, ord=self.dist_norm, dim=-1)
            # density-aware distance
            reward_particle_mask = torch.logical_or(goal_mask[:, i], latent_dist_mask_1)
            particles_for_density = torch.ones_like(min_indices)
            particles_for_density[reward_particle_mask] = 0
            count_1 = torch.zeros_like(min_indices).scatter_add_(1, min_indices, particles_for_density)
            weight_1 = (count_1.gather(1, min_indices).float() + 1e-6) ** (-1)
            dist_1 = weight_1 * dist_1
            # filter particles' contribution to reward based on mask
            dist_1[reward_particle_mask] = 0
            # check for existence of particles with no match
            unmatched_particle = torch.any(torch.logical_and(~goal_mask[:, i], latent_dist_mask_1), dim=-1).long()
            # compute number of particle groups for normalization
            n_particle_groups = torch.sum(count_1 > 0, 1) + unmatched_particle
            # make sure we don't divide by zero
            n_particle_groups = torch.maximum(n_particle_groups, torch.ones_like(n_particle_groups))
            # compute reward
            reward_g2s = - (torch.sum(dist_1, 1) + unmatched_particle) / n_particle_groups

            # State -> Goal distance
            # compute distance to closest matching particle
            min_dists, min_indices = torch.min(P, 2)
            latent_dist_mask_2 = min_dists > self.latent_dist_threshold
            a_goal = state_particles
            d_goal = torch.gather(goal_particles, 1, min_indices.unsqueeze(-1).expand(-1, -1, goal_particles.shape[-1]))
            dist_2 = torch.linalg.norm(a_goal - d_goal, ord=self.dist_norm, dim=-1)
            # density-aware distance
            reward_particle_mask = torch.logical_or(state_mask[:, i], latent_dist_mask_2)
            particles_for_density = torch.ones_like(min_indices)
            particles_for_density[reward_particle_mask] = 0
            count_2 = torch.zeros_like(min_indices).scatter_add_(1, min_indices, particles_for_density)
            weight_2 = (count_2.gather(1, min_indices).float() + 1e-6) ** (-1)
            dist_2 = weight_2 * dist_2
            # filter particles' contribution to reward based on mask
            dist_2[reward_particle_mask] = 0
            # check for existence of particles with no match
            unmatched_particle = torch.any(torch.logical_and(~state_mask[:, i], latent_dist_mask_2), dim=-1).long()
            # compute number of particle groups for normalization
            n_particle_groups = torch.sum(count_2 > 0, 1) + unmatched_particle
            # make sure we don't divide by zero
            n_particle_groups = torch.maximum(n_particle_groups, torch.ones_like(n_particle_groups))
            # compute reward
            reward_s2g = - (torch.sum(dist_2, 1) + unmatched_particle) / n_particle_groups

            # take mean over chamfer rewards
            reward_view[i] = (reward_g2s + reward_s2g) / 2
            # set max negative reward in case there are no valid particles in either state or goal
            no_valid_particles = torch.logical_or(torch.all(goal_mask[:, i], dim=-1), torch.all(state_mask[:, i], dim=-1))
            reward_view[i][no_valid_particles] = -1

        reward = torch.mean(reward_view, dim=0)

        return self.reward_scale * reward.unsqueeze(-1)

    def get_particle_mask(self, particles):

        if len(particles.shape) == 4:
            bs, n_views, n_particles, feature_dim = particles.shape
        else:
            bs, n_particles, feature_dim = particles.shape
            n_views = 1

        pixel_xy = particles[..., :2]
        scale_xy = particles[..., 2:4]
        vis_features = particles[..., 5:9]
        obj_on = particles[..., -1]

        # latent classifier condition
        if self.latent_classifier is not None:
            obj_class_cond = self.latent_classifier.classify(vis_features)
        else:
            obj_class_cond = torch.ones_like(obj_on)

        mask = (obj_class_cond == 0)

        return mask


if __name__ == '__main__':
    """
    This script can be used to debug the Chamfer Reward.
    To plot goal-dependent reward visualization, set vis_reward=True.
    """

    # config
    config = yaml.safe_load(Path('config/n_cubes/Config.yaml').read_text())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    density_aware = True
    vis_reward = True
    num_objects = 1
    npy_data_path = f'<data_path>.npy'
    dlp_dir_path = './latent_rep_chkpts/dlp_push_5C'
    normalizer_pkl_path = f'results/normalizer_panda_push_{num_objects}C.pkl'
    latent_classifier_chkpt_path = f'latent_classifier_chkpts/latent_classifier_{num_objects}C_dlp_push_5C'

    # load and organize image data
    loaded_data = np.load(npy_data_path)
    if len(loaded_data.shape) == 6:
        n_episodes, horizon, n_views, c, h, w = loaded_data.shape
    else:
        n_episodes, horizon, c, h, w = loaded_data.shape
        n_views = 1
    img_data = np.random.permutation(loaded_data.reshape([-1, n_views, c, h, w]))

    # load dlp model
    dlp = load_pretrained_rep_model(dlp_dir_path, model_type='dlp').to(device)

    # load dlp normalizer
    if os.path.exists(normalizer_pkl_path):
        with open(normalizer_pkl_path, 'rb') as file:
            dlp_normalizer = pickle.load(file)
        print(f"Loaded DLP normalizer from {normalizer_pkl_path}")

    else:
        print(f"Calculating stats for DLP normalizer...")
        obs = torch.tensor(img_data[0], device=device)
        particles = convert_obs_to_dlp_rep(obs.unsqueeze(0), dlp)
        dlp_normalizer = RMSNormalizer(epsilon=1e-6, shape=particles.shape[-1])

        dl = DataLoader(img_data, batch_size=32, shuffle=False)
        with torch.no_grad():
            for batch in tqdm(dl):
                obs = batch.to(device)
                particles = convert_obs_to_dlp_rep(obs, dlp)
                dlp_normalizer.update(particles.cpu().numpy().reshape(-1, particles.shape[-1]))
        with open(normalizer_pkl_path, 'wb') as file:
            pickle.dump(dlp_normalizer, file)
        print(f"Saved DLP normalizer in {normalizer_pkl_path}")

    # load latent classifier
    latent_classifier = MLPClassifier(**config['Reward']['LatentClassifier']['params'])
    latent_classifier.mlp.load_state_dict(torch.load(latent_classifier_chkpt_path))
    print(f"Loaded latent_classifier model from {latent_classifier_chkpt_path}")

    # initialize reward model
    if density_aware:
        DensityAwareChamferReward(particle_normalizer=dlp_normalizer,
                                  latent_classifier=latent_classifier,
                                  **config['Reward']['Chamfer']).to(device)
    else:
        reward_model = ChamferReward(particle_normalizer=dlp_normalizer,
                                     latent_classifier=latent_classifier,
                                     smorl=config['Model']['method'] == 'SMORL',
                                     **config['Reward']['Chamfer']).to(device).to(device)

    # compute reward using model
    batch_size = 8
    dl = DataLoader(img_data, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in tqdm(dl):
            images = batch.to(device)
            # media = torch.cat([media[:batch_size//2], media[:batch_size//2]], dim=0)
            particles = convert_obs_to_dlp_rep(images, dlp)
            # normalize particles
            particles = dlp_normalizer.normalize(particles)
            # prepare input for reward model
            obs = {"achieved_goal": particles[:batch_size // 2],
                   "desired_goal": particles[batch_size // 2:]}
            # compute reward
            reward = reward_model(obs)
            # plot reward and dlp media
            if vis_reward:
                plot_reward(reward.squeeze(), images[:batch_size // 2], images[batch_size // 2:], dlp, reward_model)
