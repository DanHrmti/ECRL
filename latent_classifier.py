import os
import pickle
import json
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from dlp2.models import ObjectDLP


"""
Helpers
"""


def extract_dlp_features(obs, dlp_model):
    normalized_observations = obs.to(torch.float32) / 255

    with torch.no_grad():
        encoded_output = dlp_model.encode_all(normalized_observations, deterministic=True)
        particles = get_dlp_rep(encoded_output)

    return particles, encoded_output["cropped_objects"]


def get_dlp_rep(dlp_output):
    pixel_xy = dlp_output['z']
    scale_xy = dlp_output['mu_scale']
    depth = dlp_output['mu_depth']
    visual_features = dlp_output['mu_features']
    transp = dlp_output['obj_on'].unsqueeze(dim=-1)
    rep = torch.cat((pixel_xy, scale_xy, depth, visual_features, transp), dim=-1)
    return rep


def load_pretrained_dlp(dir_path):
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
    ckpt_path = os.path.join(dir_path, f'saves/panda_dlp_best.pth')
    model.load_state_dict(torch.load(ckpt_path))
    print(f"Loaded pretrained representation model from {ckpt_path}")

    model.eval()
    model.requires_grad_(False)

    return model


def get_user_tags(length, possible_tags):
    input_ok = False
    while(not input_ok):
        input_tag_string = input('Enter tags separated by space: ')
        input_tag_string = input_tag_string.split()
        tag_list = [int(tag) for tag in input_tag_string]
        # check input is valid
        if len(tag_list) != length:
            print("Wrong number of tags, please tag again...")
        elif not all(tag in possible_tags for tag in tag_list):
            print("Found invalid tag in tag list, please tag again...")
        else:
            input_ok = True
    return tag_list


def plot_particle_latent_glimpses(crops, particle_vis_features, dlp_model):
    # decode object glimpses
    dec_objects = dlp_model.fg_module.object_dec(particle_vis_features)
    dec_objects = dec_objects.unsqueeze(0)
    _, object_glimpses = torch.split(dec_objects, [1, 3], dim=-3)
    # plot glimpses
    glimpses = torch.cat([crops, object_glimpses], dim=0)
    plot_glimpses(glimpses, np.tile(np.arange(dec_objects.shape[1]).reshape([1, -1]), (2, 1)))


def plot_glimpses(dec_object_glimpses, idx, save_dir=None):
    B, N, C, H, W = dec_object_glimpses.shape
    n_row, n_col = 1, B

    fig = plt.figure(figsize=(2 * n_col, 7 * n_row))
    fig.suptitle(f"Particle Glimpses", fontsize=14)

    for i in range(B):
        ax = fig.add_subplot(n_row, n_col, i+1)
        glimpses = dec_object_glimpses[i]
        glimpses = torch.cat([glimpses[i] for i in range(len(glimpses))], dim=-2)
        glimpses = glimpses.detach().cpu().numpy()
        glimpses = np.moveaxis(glimpses, 0, -1)
        ax.imshow(glimpses)
        ax.set_xticks([], [])
        ax.set_yticks(range(W // 2 - 1, W // 2 + W * N - 1, W), [f"{idx[i][n]:1d}" for n in range(N)])
        for j in range(1, N):
            ax.axhline(y=j * W, color='black')

    fig.tight_layout()
    if save_dir is not None:
        plt.savefig(save_dir, bbox_inches='tight')
    else:
        plt.show()
    return


class MLPClassifier(nn.Module):
    def __init__(self, latent_vis_feature_dim=4, h_dim=128, n_hidden_layers=3):
        super(MLPClassifier, self).__init__()
        layers = [nn.Linear(latent_vis_feature_dim, h_dim), nn.ReLU(True)]
        for _ in range(n_hidden_layers-1):
            layers += [nn.Linear(h_dim, h_dim), nn.ReLU(True)]
        layers += [nn.Linear(h_dim, 2)]

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

    def classify(self, x):
        logits = self.mlp(x)
        return torch.argmax(logits, dim=-1)


if __name__ == '__main__':
    """
    Script for training the DLP latent binary classifier for the Chamfer Reward filter.
    """

    tag = True
    total_images = 20
    npy_data_path = '<data_path>.npy'
    dlp_dir_path = 'latent_rep_chkpts/dlp_push_5C'
    dataset_save_dir = '<tagged_data_save_dir>'
    classifier_model_ckpt_path = 'latent_classifier_chkpts/<classifier_name>'

    # network hyperparameters
    latent_vis_feature_dim = 4
    h_dim = 128
    n_hidden_layers = 3

    # training hyperparameters
    num_epochs = 20
    bs = 64
    lr = 0.001
    cross_entropy_weights = [1.0, 0.4]  # assuming tag '1' is object of interest

    train_path = os.path.join(dataset_save_dir, 'train.pkl')
    valid_path = os.path.join(dataset_save_dir, 'valid.pkl')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load dlp model
    dlp = load_pretrained_dlp(dlp_dir_path).to(device)

    #################################
    #            Tag Data           #
    #################################

    if tag:
        # create dataset dir
        os.makedirs(dataset_save_dir, exist_ok=True)

        # load and shuffle image data
        loaded_data = np.load(npy_data_path)
        n_episodes, horizon, n_views, c, h, w = loaded_data.shape
        img_data = np.random.permutation(loaded_data.reshape([-1, c, h, w]))

        num_train_images = int(0.8 * total_images)
        print(f'Number of training images: {num_train_images}')
        print(f'Number of validation images: {total_images - num_train_images}\n')

        # train-validation split
        images = img_data[:total_images]
        train_images = images[:num_train_images]
        validation_images = images[num_train_images:]

        # tag and save train data
        print(f"Tag training data please, '1' for object of interest and '0' otherwise.")
        particle_vis_feature_list, tag_list = [], []
        dl = DataLoader(train_images, batch_size=1, shuffle=False)
        with torch.no_grad():
            for batch in tqdm(dl):
                # extract particle visual features
                obs = batch.to(device)
                particles, cropped_objects = extract_dlp_features(obs, dlp)
                particle_vis_features = particles[..., 5:9]
                # tag particle data
                plot_particle_latent_glimpses(cropped_objects, particle_vis_features, dlp)
                tags = get_user_tags(length=particle_vis_features.shape[1], possible_tags=[0, 1])
                plt.close()
                # add data and tags to list
                particle_vis_feature_list.extend(particle_vis_features.squeeze().cpu().numpy())
                tag_list.extend(tags)
        # save data
        train_data = list(zip(particle_vis_feature_list, tag_list))
        with open(train_path, 'wb') as file:
            pickle.dump(train_data, file)
        print(f"Saved tagged training data to {train_path}\n")

        # tag and save validation data
        print(f"Tag validation data please...")
        particle_vis_feature_list, tag_list = [], []
        dl = DataLoader(validation_images, batch_size=1, shuffle=False)
        with torch.no_grad():
            for batch in tqdm(dl):
                # extract particle visual features
                obs = batch.to(device)
                particles, cropped_objects = extract_dlp_features(obs, dlp)
                particle_vis_features = particles[..., 5:9]
                # tag particle data
                plot_particle_latent_glimpses(cropped_objects, particle_vis_features, dlp)
                tags = get_user_tags(length=particle_vis_features.shape[1], possible_tags=[0, 1])
                plt.close()
                # add data and tags to list
                particle_vis_feature_list.extend(particle_vis_features.squeeze().cpu().numpy())
                tag_list.extend(tags)
        # save data
        validation_data = list(zip(particle_vis_feature_list, tag_list))
        with open(valid_path, 'wb') as file:
            pickle.dump(validation_data, file)
        print(f"Saved tagged validation data to {valid_path}\n")

    #################################
    #        Train Classifier       #
    #################################

    # define network
    latent_classifier = MLPClassifier(latent_vis_feature_dim, h_dim, n_hidden_layers).to(device)

    # define criterion and optimizer
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(cross_entropy_weights, device=device))
    optimizer = optim.Adam(latent_classifier.parameters(), lr=lr)

    # load training and validation data
    with open(train_path, 'rb') as file:
        train_data = pickle.load(file)
    print(f"Loaded training data from {train_path}")

    with open(valid_path, 'rb') as file:
        valid_data = pickle.load(file)
    print(f"Loaded training data from {valid_path}")

    train_dl = DataLoader(train_data, batch_size=bs, shuffle=True)
    valid_dl = DataLoader(valid_data, batch_size=bs, shuffle=True)

    # training loop
    for epoch in range(num_epochs):
        running_loss, running_acc, num_examples = 0.0, 0.0, 0
        for batch in tqdm(train_dl):
            latent_features, labels = batch
            latent_features, labels = latent_features.to(device), labels.to(device).to(torch.long)
            # forward
            logits = latent_classifier(latent_features)
            loss = criterion(logits, labels)
            # backward + optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # gather statistics
            running_loss += loss.item() * len(batch)
            running_acc += torch.sum(torch.argmax(logits, dim=-1) == labels)
            num_examples += len(labels)
        # calculate validation stats
        valid_running_loss, valid_running_acc, valid_num_examples = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in tqdm(valid_dl):
                latent_features, labels = batch
                latent_features, labels = latent_features.to(device), labels.to(device).to(torch.long)
                # forward
                logits = latent_classifier(latent_features)
                loss = criterion(logits, labels)
                # gather statistics
                valid_running_loss += loss.item() * len(batch)
                valid_running_acc += torch.sum(torch.argmax(logits, dim=-1) == labels)
                valid_num_examples += len(labels)
        # print epoch statistics
        print(f'\nEpoch {epoch} Stats')
        print(f'Training loss: {running_loss / num_examples:.3f}, accuracy: {running_acc / num_examples:.3f}')
        print(f'Validation loss: {valid_running_loss / valid_num_examples:.3f}, accuracy: {valid_running_acc / valid_num_examples:.3f}')

    print('\nFinished Training')

    # save classifier
    torch.save(latent_classifier.mlp.state_dict(), classifier_model_ckpt_path)
    print(f"Latent classifier model saved in {classifier_model_ckpt_path}")



