from torchvision import models
import torch
import torch.nn as nn
import timm


def convert_to_inplace_relu(model):
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = True


def compute_loss(x1, x2, bw, weight):
    mask = nn.functional.interpolate(bw, size=x1.shape[2:]).squeeze()
    # return weight * torch.mean(torch.mean(mask * torch.mean(torch.abs(x1 - x2), 1, True), 2), 2)
    return weight * torch.mean(torch.mean(mask * torch.mean(torch.abs(x1 - x2), 1).unsqueeze(1), 2), 2)


class ResNet(nn.Module):
    def __init__(self, slug, pretrained=True):
        super().__init__()
        if not pretrained:
            print("Caution, not loading pretrained weights.")

        if slug == 'r18':
            self.resnet = models.resnet18(pretrained=pretrained)
        elif slug == 'r34':
            self.resnet = models.resnet34(pretrained=pretrained)
        elif slug == 'r50':
            self.resnet = models.resnet50(pretrained=pretrained)
        elif slug == 'r101':
            self.resnet = models.resnet101(pretrained=pretrained)
        elif slug == 'r152':
            self.resnet = models.resnet152(pretrained=pretrained)
        elif slug == 'rx50':
            self.resnet = models.resnext50_32x4d(pretrained=pretrained)
        elif slug == 'rx101':
            self.resnet = models.resnext101_32x8d(pretrained=pretrained)
        elif slug == 'r50d':
            self.resnet = timm.create_model('gluon_resnet50_v1d',
                                            pretrained=pretrained)
            convert_to_inplace_relu(self.resnet)
        elif slug == 'r101d':
            self.resnet = timm.create_model('gluon_resnet101_v1d',
                                            pretrained=pretrained)
            convert_to_inplace_relu(self.resnet)

        else:
            assert False, "Bad slug: %s" % slug

        self.extra = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
        )
        self.feat_layers = [self.resnet.maxpool,
                            self.resnet.layer1,
                            self.resnet.layer2,
                            self.resnet.layer3,
                            self.resnet.layer4]

    def forward(self, x1, x2, bw, weigths):
        loss = compute_loss(x1, x2, bw, weigths[0])
        x1, x2 = self.extra(x1), self.extra(x2)
        for i in range(len(self.feat_layers)):
            x1 = self.feat_layers[i](x1)
            x2 = self.feat_layers[i](x2)
            loss += compute_loss(x1, x2, bw, weigths[i + 1])
        return loss


class VGG(nn.Module):
    def __init__(self, slug, pretrained=True):
        super().__init__()
        if not pretrained:
            print("Caution, not loading pretrained weights.")

        if slug == 'v16':
            self.vgg = models.vgg16(pretrained=pretrained)
            self.feats_idx = [3, 8, 13, 22, 29]
        elif slug == 'v19':
            self.vgg = models.vgg19(pretrained=pretrained)
            self.feats_idx = [3, 8, 13, 22, 33]
        elif slug == 'v16bn':
            self.vgg = models.vgg16_bn(pretrained=pretrained)
            self.feats_idx = [5, 12, 19, 30, 42]
        elif slug == 'v19bn':
            self.vgg = models.vgg19_bn(pretrained=pretrained)
            self.feats_idx = [5, 12, 19, 30, 51]

        else:
            assert False, "Bad slug: %s" % slug

        self.vgg = self.vgg.features

    def forward(self, x1, x2, bw, weigths):
        weight_idx = 0
        loss = compute_loss(x1, x2, bw, weigths[weight_idx])
        weight_idx += 1
        for step, layer in enumerate(self.vgg):
            x1 = layer(x1)
            x2 = layer(x2)
            if step in self.feats_idx:
                loss += compute_loss(x1, x2, bw, weigths[weight_idx])
                weight_idx += 1
        return loss  # [9, 1]
