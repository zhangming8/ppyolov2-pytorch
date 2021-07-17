import torch
import torch.nn as nn
from mmcv.runner import BaseModule


# reference https://github.com/yjh0410/CSPDarkNet53
# modify by MingZhang


class Conv(nn.Module):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1, act=True):
        super(Conv, self).__init__()
        if act:
            self.convs = nn.Sequential(
                nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=False),
                nn.BatchNorm2d(c2),
                nn.LeakyReLU(0.1, inplace=True)
            )
        else:
            self.convs = nn.Sequential(
                nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=False),
                nn.BatchNorm2d(c2)
            )

    def forward(self, x):
        return self.convs(x)


class ResidualBlock(nn.Module):
    """
    basic residual block for CSP-Darknet
    """

    def __init__(self, in_ch):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv(in_ch, in_ch, k=1)
        self.conv2 = Conv(in_ch, in_ch, k=3, p=1, act=False)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        h = self.conv2(self.conv1(x))
        out = self.act(x + h)

        return out


class CSPStage(nn.Module):
    def __init__(self, c1, n=1):
        super(CSPStage, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k=1)
        self.cv2 = Conv(c1, c_, k=1)
        self.res_blocks = nn.Sequential(*[ResidualBlock(in_ch=c_) for _ in range(n)])
        self.cv3 = Conv(2 * c_, c1, k=1)

    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.res_blocks(self.cv2(x))

        return self.cv3(torch.cat([y1, y2], dim=1))


class CSPDarknet53(BaseModule):
    """
    CSPDarknet_53.
    """

    def __init__(self, depth=53, out_indices=(3, 4, 5), init_cfg=None, pretrained=None):
        super(CSPDarknet53, self).__init__()

        self.layer_1 = nn.Sequential(
            Conv(3, 32, k=3, p=1),
            Conv(32, 64, k=3, p=1, s=2),
            CSPStage(c1=64, n=1)  # p1/2
        )
        self.layer_2 = nn.Sequential(
            Conv(64, 128, k=3, p=1, s=2),
            CSPStage(c1=128, n=2)  # P2/4
        )
        self.layer_3 = nn.Sequential(
            Conv(128, 256, k=3, p=1, s=2),
            CSPStage(c1=256, n=8)  # P3/8
        )
        self.layer_4 = nn.Sequential(
            Conv(256, 512, k=3, p=1, s=2),
            CSPStage(c1=512, n=8)  # P4/16
        )
        self.layer_5 = nn.Sequential(
            Conv(512, 1024, k=3, p=1, s=2),
            CSPStage(c1=1024, n=4)  # P5/32
        )
        assert depth == 53
        self.out_indices = out_indices

        assert not (init_cfg and pretrained), 'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        out = []
        for idx, layer in enumerate([self.layer_1, self.layer_2, self.layer_3, self.layer_4, self.layer_5]):
            x = layer(x)
            if idx + 1 in self.out_indices:
                out.append(x)
        return out


class CSPDarknetSlim(BaseModule):
    """
    CSPDarknet_Slim.
    """

    def __init__(self, depth=53, out_indices=(3, 4, 5), init_cfg=None, pretrained=None):
        super(CSPDarknetSlim, self).__init__()

        self.layer_1 = nn.Sequential(
            Conv(3, 32, k=3, p=1),
            Conv(32, 64, k=3, p=1, s=2),
            CSPStage(c1=64, n=1)  # p1/2
        )
        self.layer_2 = nn.Sequential(
            Conv(64, 128, k=3, p=1, s=2),
            CSPStage(c1=128, n=1)  # P2/4
        )
        self.layer_3 = nn.Sequential(
            Conv(128, 256, k=3, p=1, s=2),
            CSPStage(c1=256, n=1)  # P3/8
        )
        self.layer_4 = nn.Sequential(
            Conv(256, 512, k=3, p=1, s=2),
            CSPStage(c1=512, n=1)  # P4/16
        )
        self.layer_5 = nn.Sequential(
            Conv(512, 1024, k=3, p=1, s=2),
            CSPStage(c1=1024, n=1)  # P5/32
        )

        assert depth == 53
        self.out_indices = out_indices

        assert not (init_cfg and pretrained), 'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        out = []
        for idx, layer in enumerate([self.layer_1, self.layer_2, self.layer_3, self.layer_4, self.layer_5]):
            x = layer(x)
            if idx + 1 in self.out_indices:
                out.append(x)
        return out


if __name__ == '__main__':
    from thop import profile

    img = torch.randn(1, 3, 416, 416)

    model = CSPDarknet53(depth=53, out_indices=(3, 4, 5))
    model.init_weights()
    # model = CSPDarknetSlim(depth=53, out_indices=(3, 4, 5))
    output = model(img)
    for o in output:
        print("output:", o.shape)
    flop, param = profile(model, (img,))
    print("flop {}, param {}".format(flop, param))
