import os
from glob import glob

import torch
import pytorch_lightning as pl
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from slot_attention.utils import rescale

class PandaPush(Dataset):
    def __init__(self, root, phase, neg_1_to_pos_1_scale: bool = False, image_size: int = 128):
        assert phase in ["train", "val", "test"]
        if phase == 'val' or phase == "test":
            phase = "valid"
        self.neg_1_to_pos_1_scale = neg_1_to_pos_1_scale

        self.root = os.path.join(root, phase)
        self.image_size = image_size
        self.mode = phase
        self.sample_length = 1

        # Get all numbers
        self.folders = []
        for file in os.listdir(self.root):
            try:
                self.folders.append(int(file))
            except ValueError:
                continue
        self.folders.sort()

        self.episodes = []
        self.EP_LEN = 50
        self.seq_per_episode = self.EP_LEN - self.sample_length + 1

        for f in self.folders:
            dir_name = os.path.join(self.root, str(f))
            paths = list(glob(os.path.join(dir_name, '*.png')))
            # if len(paths) != self.EP_LEN:
            #     continue
            # assert len(paths) == self.EP_LEN, 'len(paths): {}'.format(len(paths))
            get_num = lambda x: int(os.path.splitext(os.path.basename(x))[0])
            paths.sort(key=get_num)
            self.episodes.append(paths)
        # self.episodes = self.episodes[:4]
        if self.mode == "valid":
            self.episodes = self.episodes[:100]
        print(f'{self.mode} data: {len(self.episodes) * self.seq_per_episode}')

    def __getitem__(self, index):
        imgs = []
        # Implement continuous indexing
        ep = index // self.seq_per_episode
        offset = index % self.seq_per_episode
        end = offset + self.sample_length

        e = self.episodes[ep]
        for image_index in range(offset, end):
            img = Image.open(os.path.join(e[image_index]))
            img = img.resize((self.image_size, self.image_size))
            img = transforms.ToTensor()(img)[:3]
            imgs.append(img)

        img = torch.stack(imgs, dim=0).float().squeeze()
        if self.neg_1_to_pos_1_scale:
            img = rescale(img)

        return img

    def __len__(self):
        length = len(self.episodes)
        return length * self.seq_per_episode


class PandaPushDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_root: str,
            train_batch_size: int,
            val_batch_size: int,
            num_workers: int,
            neg_1_to_pos_1_scale: bool = False,
    ):
        super().__init__()
        self.data_root = data_root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.neg_1_to_pos_1_scale = neg_1_to_pos_1_scale

        self.train_dataset = PandaPush(
            root=self.data_root,
            phase="train",
            neg_1_to_pos_1_scale=neg_1_to_pos_1_scale,
        )
        self.val_dataset = PandaPush(
            root=self.data_root, phase="val", neg_1_to_pos_1_scale=neg_1_to_pos_1_scale
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
