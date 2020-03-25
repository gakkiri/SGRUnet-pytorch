from torchvision import models
import torch
import torch.nn as nn
import timm


def convert_to_inplace_relu(model):
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = True


class Conv1x1(nn.Module):
    def __init__(self, num_in, num_out):
        super().__init__()
        self.block = nn.Conv2d(num_in, num_out, kernel_size=1, bias=False)

    def forward(self, x):
        return self.block(x)


class Conv3x3(nn.Module):
    def __init__(self, num_in, num_out):
        super().__init__()
        self.block = nn.Conv2d(num_in, num_out, kernel_size=3, padding=1,
                               bias=False)

    def forward(self, x):
        return self.block(x)


class D(nn.Module):
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

    def forward(self, x):
        # Bottom-up pathway, from ResNet
        size = x.size()
        assert size[-1] % 32 == 0 and size[-2] % 32 == 0, \
            "image resolution has to be divisible by 32 for resnet"

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)

        d_out_1 = self.resnet.maxpool(x)
        d_out_2 = self.resnet.layer1(d_out_1)
        d_out_3 = self.resnet.layer2(d_out_2)
        d_out_4 = self.resnet.layer3(d_out_3)
        d_out_5 = self.resnet.layer4(d_out_4)
        return d_out_1, d_out_2, d_out_3, d_out_4, d_out_5


if __name__ == '__main__':
    d = D(slug='r18').eval()

    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        d_outs = d(x)
    for out in d_outs:
        print(out.shape)