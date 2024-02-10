import os
import numpy as np
from PIL import Image
from tqdm import tqdm

path_to_npy = '<data_path>.npy'
data_save_dir = '<dataset_dir>'

# load raw image data
loaded_file = np.load(path_to_npy)
if len(loaded_file.shape) == 6:
    n_episodes, horizon, n_views, c, h, w = loaded_file.shape
    loaded_file = loaded_file.reshape([n_episodes, -1, c, h, w])
print(f"Processing data from: {path_to_npy}")

# create dirs for saving dataset
train_dir = os.path.join(data_save_dir, 'train')
valid_dir = os.path.join(data_save_dir, 'valid')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)
print(f"Saving processed data in: {data_save_dir}")

# random permutation
loaded_file = np.random.permutation(loaded_file)

total_episodes = loaded_file.shape[0]
num_train_ep = int(0.85 * total_episodes)
print(f'num_train_ep: {num_train_ep}')

for ep in tqdm(range(loaded_file.shape[0])):
    if ep < num_train_ep:
        ep_dir = os.path.join(train_dir, str(ep))
    else:
        ep_dir = os.path.join(valid_dir, str(ep))

    os.makedirs(ep_dir, exist_ok=True)
    for i in range(loaded_file.shape[1]):
        im = loaded_file[ep][i].transpose(1, 2, 0)
        Image.fromarray(im).save(os.path.join(ep_dir, f'{i}.png'))