import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from einops import rearrange
import torch.nn.functional as F
from model.conv_layers import BasicBlock, ConvNormAct, DepthwiseSeparableConv, SimAM_Block
from model.trans_layers import TransformerBlock, LayerNorm
import numpy as np
import math


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, S, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, window_size, C)
    """
    B, S, H, W, C = x.shape
    # x = x.view(B, S // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
    # windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0], window_size[1], window_size[2], C)

    windows = rearrange(x, 'b (s p1) (h p2) (w p3) c -> (b s h w) p1 p2 p3 c',
                        p1=window_size[0], p2=window_size[1], p3=window_size[2], c=C)
    return windows


def window_reverse(windows, window_size, S, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, window_size, C)
        window_size (int): Window size
        S (int): Slice of image
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, S ,H, W, C)
    """
    B = int(windows.shape[0] / (S * H * W /
                                window_size[0] / window_size[1] / window_size[2]))
    # x = windows.view(B, S // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    # x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, S, H, W, -1)
    x = rearrange(windows, '(b s h w) p1 p2 p3 c -> b (s p1) (h p2) (w p3) c',
                  p1=window_size[0], p2=window_size[1], p3=window_size[2], b=B,
                  s=S // window_size[0], h=H // window_size[1], w=W // window_size[2])
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Ws, Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                        num_heads))  # 2*Ws-1 * 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_s = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(
            [coords_s, coords_h, coords_w]))  # 3, Ws, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Ws*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - \
                          coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - \
                                    1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * \
                                    (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index",
                             relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size  # 0 or window_size // 2
        self.mlp_ratio = mlp_ratio
        # if min(self.input_resolution) <= min(self.window_size):  # 56,28,14,7 >= 7
        #     # if window size is larger than input resolution, we don't partition windows
        #     self.shift_size = 0
        #     self.window_size = min(self.input_resolution)
        if self.shift_size != 0:
            assert 0 <= min(self.shift_size) < min(
                self.window_size), "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        if max(self.shift_size) > 0:
            # calculate attention mask for SW-MSA
            S, H, W = self.input_resolution
            img_mask = torch.zeros((1, S, H, W, 1))  # 1 S H W 1
            s_slices = (slice(0, -self.window_size[0]),
                        slice(-self.window_size[0], -self.shift_size[0]),
                        slice(-self.shift_size[0], None))
            h_slices = (slice(0, -self.window_size[1]),
                        slice(-self.window_size[1], -self.shift_size[1]),
                        slice(-self.shift_size[1], None))
            w_slices = (slice(0, -self.window_size[2]),
                        slice(-self.window_size[2], -self.shift_size[2]),
                        slice(-self.shift_size[2], None))
            cnt = 0
            for s in s_slices:
                for h in h_slices:
                    for w in w_slices:
                        img_mask[:, s, h, w, :] = cnt
                        cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(
                -1, self.window_size[0] * self.window_size[1] * self.window_size[2])
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        s, h, w = self.input_resolution
        B, C, S, H, W = x.shape
        assert S == s and H == h and W == w, "input feature has wrong size"
        x = rearrange(x, 'b c s h w -> b (s h w) c')
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, S, H, W, C)

        # cyclic shift
        if max(self.shift_size) > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
        else:
            shifted_x = x
        # partition windows
        # nW*B, window_size, window_size, C
        x_windows = window_partition(shifted_x, self.window_size)
        # nW*B, window_size*window_size*window_size, C
        x_windows = x_windows.view(
            -1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)

        # W-MSA/SW-MSA
        # nW*B, window_size*window_size, C
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # merge windows
        attn_windows = attn_windows.view(
            -1, self.window_size[0], self.window_size[1], self.window_size[2], C)
        shifted_x = window_reverse(
            attn_windows, self.window_size, S, H, W)  # B H' W' C

        # reverse cyclic shift
        if max(self.shift_size) > 0:
            x = torch.roll(shifted_x, shifts=(
                self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x
        # FFN
        x = x.view(B, S * H * W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = rearrange(x, 'b (s h w) c -> b c s h w', s=S, h=H, w=W)

        return x





class SemanticMapFusion(nn.Module):
    def __init__(self, in_dim_list, dim, heads, depth=1, norm=nn.BatchNorm3d, attn_drop=0., proj_drop=0., ps=False):
        super().__init__()

        self.dim = dim
        self.ps = ps
        # project all maps to the same channel num
        self.in_proj = nn.ModuleList([])
        self.map_size = [8, 8, 8]

        for i in range(len(in_dim_list)):
            self.in_proj.append(nn.Sequential(
                nn.AdaptiveAvgPool3d(self.map_size),
                nn.Conv3d(in_dim_list[i], dim, kernel_size=1, bias=False)
            ))


        if self.ps == True:
            self.positional_encoding = nn.ModuleList([])
            for i in range(len(in_dim_list)):
                self.positional_encoding.append(PositionalEncoding3D(dim))

        self.fusion = TransformerBlock(dim, depth, heads, dim // heads, dim, attn_drop=attn_drop, proj_drop=proj_drop)

        # project all maps back to their origin channel num
        self.out_proj = nn.ModuleList([])
        for i in range(len(in_dim_list)):
            self.out_proj.append(nn.Conv3d(dim, in_dim_list[i], kernel_size=1, bias=False))


        self.map_reduction1 = nn.Conv3d(in_dim_list[0] * 3, in_dim_list[0], kernel_size=1, bias=False)
        self.map_reduction2 = nn.Conv3d(in_dim_list[1] * 3, in_dim_list[1], kernel_size=1, bias=False)
        self.map_reduction3 = nn.Conv3d(in_dim_list[2] * 3, in_dim_list[2], kernel_size=1, bias=False)


    # def forward(self, map_list_):
    def forward(self, map_list_, f_map_list_):

        map_list = [map_list_[1], map_list_[2], map_list_[3]]
        B, _, _, _, _ = map_list[0].shape

        if self.ps == True:
            p1 = self.in_proj[0](map_list[0])
            p1 = p1 + self.positional_encoding[0](p1)
            p2 = self.in_proj[1](map_list[1])
            p2 = p2 + self.positional_encoding[1](p2)
            p3 = self.in_proj[2](map_list[2])
            p3 = p3 + self.positional_encoding[2](p3)

            proj_maps = [p1.view(B, self.dim, -1).permute(0, 2, 1), p2.view(B, self.dim, -1).permute(0, 2, 1), p3.view(B, self.dim, -1).permute(0, 2, 1)]

        else:
            proj_maps = [self.in_proj[i](map_list[i]).view(B, self.dim, -1).permute(0, 2, 1) for i in range(len(map_list))]

        # B, L, C where L=DHW
        proj_maps = torch.cat(proj_maps, dim=1)
        attned_maps = self.fusion(proj_maps)

        attned_maps = attned_maps.chunk(len(map_list), dim=1)
        maps_out = [map_list_[0]] + [self.out_proj[i](
            attned_maps[i].permute(0, 2, 1).view(B, self.dim, self.map_size[0], self.map_size[1], self.map_size[
                2])) for i in range(len(map_list))]

        #
        # maps_out[1] = self.map_reduction1(torch.cat([map_list[0], F.interpolate(maps_out[1], size=map_list[0].shape[-3:], mode='trilinear',
        #                                                                            align_corners=True)], dim=1))
        #
        # maps_out[2] = self.map_reduction2(torch.cat([map_list[1], F.interpolate(maps_out[2], size=map_list[1].shape[-3:], mode='trilinear',
        #                                                                            align_corners=True)], dim=1))
        # maps_out[3] = self.map_reduction3(torch.cat([map_list[2], F.interpolate(maps_out[3], size=map_list[2].shape[-3:], mode='trilinear',
        #                                                                            align_corners=True)], dim=1))

        maps_out[1] = self.map_reduction1(torch.cat([map_list[0], f_map_list_[1], F.interpolate(maps_out[1], size=map_list[0].shape[-3:], mode='trilinear',
                                                                                   align_corners=True)], dim=1))

        maps_out[2] = self.map_reduction2(torch.cat([map_list[1], f_map_list_[2], F.interpolate(maps_out[2], size=map_list[1].shape[-3:], mode='trilinear',
                                                                                   align_corners=True)], dim=1))
        maps_out[3] = self.map_reduction3(torch.cat([map_list[2], f_map_list_[3], F.interpolate(maps_out[3], size=map_list[2].shape[-3:], mode='trilinear',
                                                                                   align_corners=True)], dim=1))
        return maps_out


class PositionalEncoding3D(nn.Module):

    def __init__(self, channels):

        """

        :param channels: The last dimension of the tensor you want to apply pos emb to.

        """

        super(PositionalEncoding3D, self).__init__()

        channels = int(np.ceil(channels / 6) * 2)

        if channels % 2:
            channels += 1

        self.channels = channels

        inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float() / channels))

        self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):

        """

        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)

        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)

        """
        # tensor: b, c, d, h, w ->
        tensor = rearrange(tensor, 'b c d h w -> b d h w c')
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")

        batch_size, x, y, z, orig_ch = tensor.shape

        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())

        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())

        pos_z = torch.arange(z, device=tensor.device).type(self.inv_freq.type())

        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)

        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)

        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)

        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1).unsqueeze(1)

        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1).unsqueeze(1)

        emb_z = torch.cat((sin_inp_z.sin(), sin_inp_z.cos()), dim=-1)

        emb = torch.zeros((x, y, z, self.channels * 3), device=tensor.device).type(tensor.type())

        emb[:, :, :, :self.channels] = emb_x

        emb[:, :, :, self.channels:2 * self.channels] = emb_y

        emb[:, :, :, 2 * self.channels:] = emb_z
        emb = emb[None, :, :, :, :orig_ch].repeat(batch_size, 1, 1, 1, 1)
        emb = rearrange(emb, 'b d h w c -> b c d h w')

        return emb


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=[3, 3, 3], stride=1, block=BasicBlock, norm=nn.BatchNorm3d, act=nn.GELU):
        super().__init__()

        pad_size = [i // 2 for i in kernel_size]
        # self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=pad_size, bias=False)
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=[7,7,7], stride=(1,1,1), padding=3, bias=False)
        self.conv2 = SimAM_Block(out_ch, out_ch, kernel_size=kernel_size, norm=norm, act=act)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        return out

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=[3, 3, 3], stride=1, block=BasicBlock, norm=nn.BatchNorm3d, act=nn.GELU):
        super().__init__()

        pad_size = [i // 2 for i in kernel_size]
        self.conv1 = block(in_ch, in_ch, kernel_size=kernel_size, norm=norm, act=act)
        # self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=[7,7,7], stride=(1,1,1), padding=3, bias=False)
        self.conv2 = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x):
        x = F.interpolate(x, size=(64, 128, 128), mode='trilinear', align_corners=True)
        out = self.conv1(x)
        out = self.conv2(out)

        return out


class ECA_reduction(nn.Module):
    def __init__(self, n_channels, k_size=3, gamma=2, b=1):
        super(ECA_reduction, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        # https://github.com/BangguWu/ECANet/issues/243
        # dynamically computing the k_size
        t = int(abs((math.log(n_channels, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1

        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.reduction = nn.Conv3d(n_channels*2, n_channels, kernel_size=1, bias=False)

    def forward(self, x, f):
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
        out = x * y.expand_as(x)

        f_y = self.global_avg_pool(f)
        # Two different branches of ECA modul
        f_y = self.conv(f_y.squeeze(-1).squeeze(-2).transpose(-2, -1)).transpose(-2, -1).unsqueeze(-2).unsqueeze(
            -1)  # b, c, d, h, w = x.size()

        # Multi-scale information fusion
        f_y = self.sigmoid(f_y)
        f_out = f * f_y.expand_as(f)

        out = self.reduction(torch.concat([out, f_out], dim=1))
        return out

class ECA_reduction_out(nn.Module):
    def __init__(self, n_channels, k_size=3, gamma=2, b=1):
        super(ECA_reduction_out, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        # https://github.com/BangguWu/ECANet/issues/243
        # dynamically computing the k_size
        t = int(abs((math.log(n_channels, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1

        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.reduction = nn.Conv3d(n_channels*3, n_channels, kernel_size=1, bias=False)

    def forward(self, x, f, s):
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
        out = x * y.expand_as(x)

        f_y = self.global_avg_pool(f)
        # Two different branches of ECA modul
        f_y = self.conv(f_y.squeeze(-1).squeeze(-2).transpose(-2, -1)).transpose(-2, -1).unsqueeze(-2).unsqueeze(
            -1)  # b, c, d, h, w = x.size()

        # Multi-scale information fusion
        f_y = self.sigmoid(f_y)
        f_out = f * f_y.expand_as(f)

        s_y = self.global_avg_pool(s)
        # Two different branches of ECA modul
        s_y = self.conv(s_y.squeeze(-1).squeeze(-2).transpose(-2, -1)).transpose(-2, -1).unsqueeze(-2).unsqueeze(
            -1)  # b, c, d, h, w = x.size()

        # Multi-scale information fusion
        s_y = self.sigmoid(s_y)
        s_out = s * s_y.expand_as(s)

        out = self.reduction(torch.concat([out, f_out, s_out], dim=1))
        return out