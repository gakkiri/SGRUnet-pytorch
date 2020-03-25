from torchvision import transforms
import random
from PIL import Image
from config import config


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img_rgb, img_bw):
        flip = False
        if random.random() < self.p:
            img_rgb = img_rgb.transpose(Image.FLIP_LEFT_RIGHT)
            img_bw = img_bw.transpose(Image.FLIP_LEFT_RIGHT)
            flip = True

        return img_rgb, img_bw, flip


def get_tsfm():
    tsfm = [
        transforms.ToTensor(),
    ]
    if config.train_size is not None:
        tsfm.insert(0, transforms.Resize((config.train_size, config.train_size)))

    return transforms.Compose(tsfm)
