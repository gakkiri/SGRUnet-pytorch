from model.layernorm import *
from config import config

if config.bn:
    Norm = nn.BatchNorm2d
else:
    Norm = LayerNorm


class Conv2DLReLU(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, stride=1, padding=0):
        super().__init__()
        # NOTE: kernel_size=2, stride=1, and padding='SAME' in the paper code
        self.conv = nn.Conv2d(inc, outc, kernel_size, stride, padding)
        self.ln = Norm(outc)
        self.llr = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.llr(self.ln(self.conv(x)))


class Conv2DTransposeLReLU(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        # output_size = 2 * input_size
        self.deconv = nn.ConvTranspose2d(inc, outc, kernel_size=2, stride=2, padding=0)
        self.ln = Norm(outc)
        self.llr = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.llr(self.ln(self.deconv(x)))


class SwishMod(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.conv = nn.Conv2d(inc, outc, 3, 1, 1)
        self.ln = Norm(outc)

    def forward(self, x):
        _x = torch.sigmoid(self.ln(self.conv(x)))
        return x.mul(_x)


class SwishGatedBlock(nn.Module):
    def __init__(self, inc, outc, cat=False, conv1x1=True):
        super().__init__()
        self.conv1x1 = conv1x1

        if conv1x1:
            self.conv0 = Conv2DLReLU(inc, outc, padding=1)
            inc = outc
            self.conv1 = Conv2DLReLU(inc, outc, padding=1)
        else:
            self.conv1 = Conv2DLReLU(inc, outc, padding=1)
        self.conv2 = Conv2DLReLU(outc, outc, padding=1)

        self.pooling = nn.MaxPool2d(2)
        if cat:
            self.deconv1 = Conv2DTransposeLReLU(outc, outc)
            self.deconv2 = Conv2DTransposeLReLU(inc, outc)
            self.swish_mod = SwishMod(outc, outc)
        else:
            self.swish_mod = SwishMod(inc, inc)

    def forward(self, inputs, cat=None):
        if self.conv1x1:
            inputs = self.conv0(inputs)
        x = self.conv1(inputs)
        x = self.conv2(x)

        if cat is None:
            sgb_op = self.pooling(x)
            swish = self.pooling(inputs)
            swish = self.swish_mod(swish)
            concat = [sgb_op, swish]
        else:
            sgb_op = self.deconv1(x)
            swish = self.deconv2(inputs)
            swish = self.swish_mod(swish)
            concat = [sgb_op, swish, cat]

        return torch.cat(concat, dim=1), x


class SGRU(nn.Module):
    def __init__(self, inc=1):
        super().__init__()

        down_in_channels = [inc, 97, 384, 576, 768]
        down_out_channels = [96, 192, 288, 384, 480]
        up_in_channels = [960, 1504, 1344, 1056, 768]
        up_out_channels = [512, 480, 384, 288, 192]

        self.down_conv_layer = [SwishGatedBlock(down_in_channels[0],
                                                down_out_channels[0],
                                                conv1x1=False)]
        for inch, outc in zip(down_in_channels[1:], down_out_channels[1:]):
            self.down_conv_layer.append(SwishGatedBlock(inch, outc))
        self.down_conv_layer = nn.Sequential(*self.down_conv_layer)

        self.swish_layers = []
        for channel in down_out_channels:
            self.swish_layers.append(SwishMod(channel, channel))
        self.swish_layers = nn.Sequential(*self.swish_layers)

        self.up_conv_layer = []
        for inch, outc in zip(up_in_channels, up_out_channels):
            self.up_conv_layer.append(SwishGatedBlock(inch, outc, cat=True))
        self.up_conv_layer = nn.Sequential(*self.up_conv_layer)

        self.conv1_up = nn.Sequential(
            Conv2DLReLU(down_out_channels[-1], down_out_channels[0], kernel_size=1),
            Conv2DLReLU(down_out_channels[0], down_out_channels[0], kernel_size=3, padding=1),
            Conv2DLReLU(down_out_channels[0], down_out_channels[0], kernel_size=3, padding=1),
            nn.Conv2d(down_out_channels[0], 27, kernel_size=1)
        )

    def forward(self, inp):
        convs = []
        swishs = []
        # inp: [B, 1, H, W]
        for layer in self.down_conv_layer:
            inp, conv = layer(inp)
            convs.append(conv)

        # inp: [B, 960, H/32, W/32]
        # convs: [[B, 96, H, W],
        #         [B, 192, H/2, W/2],
        #         [B, 288, H/4, W/4],
        #         [B, 384, H/8, W/8],
        #         [B, 480, H/16, W/16]]
        for i in range(len(convs)):
            swishs.append(self.swish_layers[i](convs[i]))

        # swishs: [[B, 96, H, W],
        #          [B, 192, H/2, W/2],
        #          [B, 288, H/4, W/4],
        #          [B, 384, H/8, W/8],
        #          [B, 480, H/16, W/16]]
        swishs = swishs[::-1]
        for i in range(len(self.up_conv_layer)):
            inp, _ = self.up_conv_layer[i](inp, cat=swishs[i])

        inp = self.conv1_up(inp)

        B, _, H, W = inp.shape
        inp = (inp + 1.0) / 2.0 * 255.0  # [B, 27, H, W]
        # [B, 27, H, W] -> [B, 9, 3, H, W]
        output = inp.view(B, -1, 3, H, W)
        return output
