import torch
from torch import nn


class CRNsLoss(nn.Module):
    """
    refer Photographic Image Synthesis with Cascaded Refinement Networks:
    arxiv: https://arxiv.org/abs/1707.09405
    """

    def __init__(self, config, Dnet):
        super().__init__()
        self.weights = config.loss_weight
        self.alpha = config.alpha
        self.beta = config.beta
        self.D = Dnet.eval()
        for p in self.D.parameters():
            p.requires_grad = False

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 1, 3).float().permute(0, 3, 1, 2)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 1, 3).float().permute(0, 3, 1, 2)
        if config.gpu:
            self.mean = self.mean.cuda()
            self.std = self.std.cuda()

    def normlization(self, x):
        return (x - self.mean) / self.std

    def forward(self, img_bw, img_rgb_label, imgs_rgb_gen):
        '''
        :param img_bw: [B, 1, H, W]
        :param img_rgb_label: [B, 3, H, W]
        :param imgs_rgb_gen:  [B, 9, 3, H, W]
        :return:
        '''
        losses = []
        img_rgb_label_norm = self.normlization(img_rgb_label)
        for i in range(imgs_rgb_gen.shape[1]):
            img_rgb_gen = self.normlization(imgs_rgb_gen[:, i, ...])
            losses.append(self.D(img_rgb_label_norm, img_rgb_gen, img_bw, self.weights))
        losses = torch.cat(losses).view(imgs_rgb_gen.shape[1], -1).transpose(1, 0)  # [B, 9]
        loss_min = torch.min(losses, 1)
        loss_mean = torch.mean(losses, 1)
        return (loss_min[0] * self.alpha + loss_mean * self.beta).mean()
