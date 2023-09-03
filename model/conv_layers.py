import torch
import torch.nn as nn
import math
from model.trans_layers import LayerNorm

__all__ = [
    'ConvNormAct',
    'BasicBlock',
    'Bottleneck',
    'DepthwiseSeparableConv',
    'DepthWiseConvNormAct',
    'SimAM_Block',
    'F_SimAM_Block',
]


class ConvNormAct(nn.Module):
    """
    Layer grouping a convolution, normalization and activation function
    normalization includes BN as IN
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,
                 groups=1, dilation=1, bias=False, norm=nn.BatchNorm3d, act=nn.ReLU, preact=False):

        super().__init__()
        assert norm in [nn.BatchNorm3d, nn.InstanceNorm3d, LayerNorm, True, False]
        assert act in [nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, True, False]
        # pad_size = [i // 2 for i in kernel_size]
        self.conv = nn.Conv3d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
            bias=bias
        )
        if preact:
            self.norm = norm(in_ch) if norm else nn.Identity()
        else:
            self.norm = norm(out_ch) if norm else nn.Identity()
        self.act = act() if act else nn.Identity()
        self.preact = preact

    def forward(self, x):
        if self.preact:
            out = self.conv(self.act(self.norm(x)))
        else:
            out = self.act(self.norm(self.conv(x)))

        return out


class DepthWiseConvNormAct(nn.Module):
    """
    Layer grouping a convolution, normalization and activation function
    normalization includes BN as IN
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,
                 groups=1, dilation=1, bias=False, norm=nn.BatchNorm3d, act=nn.ReLU, preact=False, eca=False):

        super().__init__()
        assert norm in [nn.BatchNorm3d, nn.InstanceNorm3d, LayerNorm, True, False]
        assert act in [nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, True, False]
        self.conv = DepthwiseSeparableConv(in_ch, out_ch, stride=stride, kernel_size=kernel_size, bias=False, fix_padding=False)
        if preact:
            self.norm = norm(in_ch) if norm else nn.Identity()
        else:
            self.norm = norm(out_ch) if norm else nn.Identity()
        self.act = act() if act else nn.Identity()
        self.preact = preact
        self.eca = eca

        self.eca_module = ECA_module(out_ch, k_size=3, gamma=2, b=1)

    def forward(self, x):

        if self.preact:
            if self.eca:
                out = self.eca_module(self.conv(self.act(self.norm(x))))
            else:
                out = self.conv(self.act(self.norm(x)))
        else:
            if self.eca:
                out = self.act(self.eca_module(self.norm(self.conv(x))))
            else:
                out = self.act(self.norm(self.conv(x)))

        return out


class SingleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=[3, 3, 3], stride=1, norm=nn.BatchNorm3d, act=nn.ReLU, preact=False):
        super().__init__()
        assert norm in [nn.BatchNorm3d, nn.InstanceNorm3d, LayerNorm, True, False]
        assert act in [nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, True, False]

        pad_size = [i // 2 for i in kernel_size]

        self.conv = ConvNormAct(in_ch, out_ch, kernel_size, stride=stride, padding=pad_size, norm=norm, act=act,
                                preact=preact)

    def forward(self, x):
        return self.conv(x)


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=[3, 3, 3], stride=1, norm=nn.BatchNorm3d, act=nn.ReLU, preact=True):
        super().__init__()
        assert norm in [nn.BatchNorm3d, nn.InstanceNorm3d, LayerNorm, True, False]
        assert act in [nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, True, False]

        pad_size = [i // 2 for i in kernel_size]

        self.conv1 = ConvNormAct(in_ch, out_ch, kernel_size, stride=stride, padding=pad_size, norm=norm, act=act,
                                 preact=preact)
        self.conv2 = ConvNormAct(out_ch, out_ch, kernel_size, stride=1, padding=pad_size, norm=norm, act=act,
                                 preact=preact)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = ConvNormAct(in_ch, out_ch, kernel_size, stride=stride, padding=pad_size, norm=norm, act=act,
                                        preact=preact)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        out += self.shortcut(residual)

        return out


class Bottleneck(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=[3, 3, 3], stride=1, groups=1, dilation=1, norm=nn.BatchNorm3d,
                 act=nn.ReLU, preact=True):
        super().__init__()
        assert norm in [nn.BatchNorm3d, nn.InstanceNorm3d, LayerNorm, True, False]
        assert act in [nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, True, False]

        pad_size = [i // 2 for i in kernel_size]

        self.expansion = 2
        self.conv1 = ConvNormAct(in_ch, out_ch // self.expansion, 1, stride=1, padding=0, norm=norm, act=act,
                                 preact=preact)
        self.conv2 = ConvNormAct(out_ch // self.expansion, out_ch // self.expansion, kernel_size, stride=stride,
                                 padding=pad_size, norm=norm, act=act, groups=groups, dilation=dilation, preact=preact)

        self.conv3 = ConvNormAct(out_ch // self.expansion, out_ch, 1, stride=1, padding=0, norm=norm, act=act,
                                 preact=preact)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = ConvNormAct(in_ch, out_ch, kernel_size, stride=stride, padding=pad_size, norm=norm, act=act,
                                        preact=preact)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        out += self.shortcut(residual)

        return out


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3, bias=False, fix_padding=False):
        super().__init__()

        if fix_padding is False:
            if isinstance(kernel_size, list):
                padding = [i // 2 for i in kernel_size]
            else:
                padding = [kernel_size // 2]
        else:
            padding = 0

        # print(padding)
        self.depthwise = nn.Conv3d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_ch,
            bias=bias
        )
        self.pointwise = nn.Conv3d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=bias
        )

    def forward(self, x):
        # print(f'depthwise input: {x.shape}')
        out = self.depthwise(x)
        # print(f'depthwise: {out.shape}')
        out = self.pointwise(out)
        # print(f'pointwise: {out.shape}')

        return out





class SEBlock(nn.Module):
    def __init__(self, in_ch, ratio=4, act=nn.ReLU):
        super().__init__()

        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excitation = nn.Sequential(
            nn.Conv3d(in_ch, in_ch // ratio, kernel_size=1),
            act(),
            nn.Conv3d(in_ch // ratio, in_ch, kernel_size=1),
            nn.Sigmoid()
        )

        self.conv = nn.Conv3d(in_ch, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, D, H, W = x.shape
        out = self.squeeze(x)
        out = self.excitation(out)

        out1 = self.conv(x)
        out1 = self.sigmoid(out1)
        out1 = torch.mul(x, out1.view(B, 1, D, H, W))
        return x * out + out1


class DropPath(nn.Module):

    def __init__(self, p=0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if (not self.p) or (not self.training):
            return x

        batch_size = x.shape[0]
        random_tensor = torch.rand(batch_size, 1, 1, 1, 1).to(x.device)
        binary_mask = self.p < random_tensor

        x = x.div(1 - self.p)
        x = x * binary_mask

        return x




class SimAM(nn.Module):
    def __init__(self, lambda_val=0.1):
        super(SimAM, self).__init__()
        self.lambda_val = lambda_val

    def forward(self, X):
        # spatial size
        n = X.shape[2] * X.shape[3] * X.shape[4] - 1

        # square of (t - u)
        d = (X - X.mean(dim=[2, 3, 4], keepdim=True)).pow(2)

        # d.sum() / n is channel variance
        v = d.sum(dim=[2, 3, 4], keepdim=True) / n

        # E_inv groups all importance of X
        E_inv = d / (4 * (v + self.lambda_val)) + 0.5

        # return attended features
        return X * torch.sigmoid(E_inv)


class SimAM_Block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=[3, 3, 3], stride=1, norm=nn.BatchNorm3d, act=nn.ReLU, preact=True):
        super().__init__()
        assert norm in [nn.BatchNorm3d, nn.InstanceNorm3d, LayerNorm, True, False]
        assert act in [nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, True, False]

        pad_size = [i // 2 for i in kernel_size]

        self.conv1 = ConvNormAct(in_ch, out_ch, kernel_size, stride=stride, padding=pad_size, norm=norm, act=act,
                                 preact=preact)
        self.conv2 = ConvNormAct(out_ch, out_ch, kernel_size, stride=1, padding=pad_size, norm=norm, act=act,
                                 preact=preact)
        self.simam = SimAM(lambda_val=0.1)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = ConvNormAct(in_ch, out_ch, kernel_size, stride=stride, padding=pad_size, norm=norm, act=act,
                                        preact=preact)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(residual)
        out = self.simam(out)

        return out


class F_SimAM_Block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=[3,3,3], stride=1, norm=nn.BatchNorm3d, act=nn.ReLU, preact=True):
        super().__init__()
        assert norm in [nn.BatchNorm3d, nn.InstanceNorm3d, LayerNorm, True, False]
        assert act in [nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, True, False]

        pad_size = [i//2 for i in kernel_size]

        self.conv1 = ConvNormAct(in_ch, out_ch//2, kernel_size, stride=stride, padding=pad_size, norm=norm, act=act, preact=preact)
        self.conv2 = ConvNormAct(out_ch//2, out_ch, kernel_size, stride=1, padding=pad_size, norm=norm, act=act, preact=preact)
        self.simam = SimAM(lambda_val=0.1)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = ConvNormAct(in_ch, out_ch, kernel_size, stride=stride, padding=pad_size, norm=norm, act=act, preact=preact)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(residual)

        out = self.simam(out)

        return out


class ECA_module(nn.Module):
    def __init__(self, n_channels, k_size=3, gamma=2, b=1):
        super(ECA_module, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        # https://github.com/BangguWu/ECANet/issues/243
        # dynamically computing the k_size
        t = int(abs((math.log(n_channels, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1

        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.global_avg_pool(x)

        # Two different branches of ECA module
        # https://github.com/BangguWu/ECANet/issues/30
        # https://github.com/BangguWu/ECANet/issues/7
        # y = self.conv(y.squeeze(-1).squeeze(-2).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-2).unsqueeze(-1) # b, c, d, h, w = x.size()
        y = self.conv(y.squeeze(-1).squeeze(-2).transpose(-2, -1)).transpose(-2, -1).unsqueeze(-2).unsqueeze(
            -1)  # b, c, d, h, w = x.size()

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)