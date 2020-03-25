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

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def normlization(self, img_rgb_label, img_rgb_gen):
        '''
        :param img_rgb_label: [B, 3, H, W], 0~1
        :param img_rgb_gen:  up
        :return: norm data
        '''
        for i in range(3):
            img_rgb_label[:, i, ...] = (img_rgb_label[:, i, ...] - self.mean[i]) / self.std[i]
            img_rgb_gen[:, i, ...] = (img_rgb_gen[:, i, ...] - self.mean[i]) / self.std[i]
        return img_rgb_label, img_rgb_gen

    def compute_loss(self, label, gen, bw, weight):
        '''
        :param label: [B, 3, H/?, W/?]
        :param gen: [B, 3, H/?, W/?]
        :param bw: [B, 1, H, W]
        :param weight:
        :return: [1]
        '''
        bw = nn.functional.interpolate(bw, size=label.shape[2:])
        B, C = label.shape[:2]
        bw, label, gen = bw.view(B, 1, -1), label.view(B, C, -1), gen.view(B, C, -1)  # [B, C, H * W]
        return weight * torch.mean(torch.mean(bw * torch.mean(torch.abs(label - gen), 1).unsqueeze(1), 2), 0)

    def forward(self, img_bw, img_rgb_label, imgs_rgb_gen):
        '''
        :param img_bw: [B, 1, H, W]
        :param img_rgb_label: [B, 3, H, W]
        :param imgs_rgb_gen:  [B, 9, 3, H, W]
        :return:
        '''
        losses = []  # [B, 9]
        for i in range(imgs_rgb_gen.shape[1]):
            img_rgb_gen = imgs_rgb_gen[:, i, ...]  # [B, 3, H, W], NOTE: data is 0~1
            img_rgb_label, img_rgb_gen = self.normlization(img_rgb_label, img_rgb_gen)

            with torch.no_grad():
                feats_rgb_label = self.D(img_rgb_label)
                feats_rgb_gen = self.D(img_rgb_gen)

            loss = self.compute_loss(img_rgb_label, img_rgb_gen, img_bw, self.weights[0])
            for feat_idx in range(len(feats_rgb_label)):
                loss += self.compute_loss(feats_rgb_label[feat_idx],
                                          feats_rgb_gen[feat_idx],
                                          img_bw, self.weights[feat_idx + 1])
            losses.append(loss)  # [1]
        losses = torch.cat(losses)  # [9]
        loss_min = torch.min(losses, 0)
        loss_mean = torch.mean(losses)

        return loss_min[0] * self.alpha + loss_mean * self.beta
