import torch
from torch.utils import data
from dataset.dataset import SketchDataset, SafebooruDataset
from trainer import Trainer
from config import config

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def main(config):
    if config.dataset == 'colorgram':
        DATASET = SketchDataset
    elif config.dataset == 'safebooru':
        DATASET = SafebooruDataset

    train_ds = DATASET(config)
    train_dl = data.DataLoader(train_ds, batch_size=config.batch_size, pin_memory=True, shuffle=True)

    val_ds = DATASET(config, train=False)
    val_dl = data.DataLoader(val_ds, batch_size=max((config.batch_size // 2), 1))

    trainer = Trainer(config, train_dl, val_dl)
    trainer.train()


if __name__ == '__main__':
    from pprint import pprint
    pprint(config)
    main(config)
