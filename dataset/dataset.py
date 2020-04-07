import os
import numpy as np
import torch.utils.data as data
from dataset.transforms import *

opj = os.path.join


class SketchDataset(data.Dataset):
    def __init__(self, config, train=True):
        if train:
            self.root = config.train_data_root
        else:
            self.root = config.val_data_root

        self.files = os.listdir(self.root)
        self.orig_size = config.orig_size
        self.train = train
        self.tsfm = get_tsfm()
        self.binary = config.binary

    def data_augment(self, img, sketch=None):
        if self.train:
            img, sketch, flip = RandomHorizontalFlip()(img, sketch)
        else:
            flip = False
        img, sketch = self.tsfm(img), self.tsfm(sketch)
        return img, sketch, flip

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        two_img = Image.open(opj(self.root, file))
        two_img = np.array(two_img)
        img = two_img[:, :self.orig_size, :]
        sketch = two_img[:, self.orig_size:, :]
        sketch = sketch[:, :, 0]

        if self.binary:
            sketch[sketch != 255] = 0

        img = Image.fromarray(img)
        sketch = Image.fromarray(sketch).convert('L')
        img, sketch, flip = self.data_augment(img, sketch)
        return img, sketch, flip


class SafebooruDataset(data.Dataset):
    def __init__(self, config, train=True):
        # './anime_colorization/data/train', './anime_colorization/data/val'
        if train:
            self.root = config.train_data_root
        else:
            self.root = config.val_data_root

        self.files = os.listdir(opj(self.root, 'img'))
        self.totensor = transforms.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        rgb = Image.open(opj(self.root, 'img', file))
        bw = Image.open(opj(self.root, 'label', file))

        rgb, bw = self.totensor(rgb), self.totensor(bw)
        return rgb, bw, False
