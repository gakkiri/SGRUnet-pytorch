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
        if config.gpu:
            self.mean = self.mean.cuda()

    def forward(self, img_bw, img_rgb_label, imgs_rgb_gen):
        '''
        :param img_bw: [B(1), 1, H, W]
        :param img_rgb_label: [B(1), 3, H, W]
        :param imgs_rgb_gen:  ×[B, 9, 3, H, W], [9, 3, H, W]
        :return:
        '''
        batch_loss = None
        for b in range(img_rgb_label.shape[0]):
            bw = img_bw[b, ...].unsqueeze(0)
            rgb_label = img_rgb_label[b, ...].unsqueeze(0)
            rgb_gen = imgs_rgb_gen[b, ...]
            losses = self.D(rgb_label - self.mean, rgb_gen - self.mean, bw, self.weights)
            loss_min = torch.min(losses, 0)
            if batch_loss is None:
                batch_loss = torch.sum(loss_min[0])*self.alpha + torch.sum(torch.mean(losses, 0))*self.beta
            else:
                batch_loss += torch.sum(loss_min[0])*self.alpha + torch.sum(torch.mean(losses, 0))*self.beta
        return batch_loss / img_rgb_label.shape[0]
