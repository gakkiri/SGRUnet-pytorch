import torch
from torch.utils import data
from dataset.dataset import SketchDataset
from trainer import Trainer
from config import config

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def main(config):
    train_ds = SketchDataset(config)
    train_dl = data.DataLoader(train_ds, batch_size=config.batch_size, pin_memory=True, shuffle=True)

    val_ds = SketchDataset(config, train=False)
    val_dl = data.DataLoader(val_ds, batch_size=max((config.batch_size // 2), 1))

    trainer = Trainer(config, train_dl, val_dl)
    trainer.train()


if __name__ == '__main__':
    main(config)
