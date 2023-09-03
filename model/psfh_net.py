import torch.utils.checkpoint as checkpoint
from model.psfh_net_utils import *
from model.conv_layers import F_SimAM_Block, SimAM_Block


class PatchMerging(nn.Module):
    """
    Modified patch merging layer that works as down-sampling
    """

    def __init__(self, dim, out_dim, norm=nn.BatchNorm3d, proj_type='linear', down_scale=[2, 2, 2], kernel_size=[3, 3, 3]):
        super().__init__()
        self.dim = dim
        assert proj_type in ['linear', 'depthwise']

        self.down_scale = down_scale

        merged_dim = 2 ** down_scale.count(2) * dim

        if proj_type == 'linear':
            self.reduction = nn.Conv3d(merged_dim, out_dim, kernel_size=1, bias=False)
        else:
            self.reduction = DepthwiseSeparableConv(merged_dim, out_dim, kernel_size=kernel_size)

        self.norm = norm(merged_dim)

    def forward(self, x):
        """
        x: B, C, D, H, W
        """
        merged_x = []
        for i in range(self.down_scale[0]):
            for j in range(self.down_scale[1]):
                for k in range(self.down_scale[2]):
                    tmp_x = x[:, :, i::self.down_scale[0], j::self.down_scale[1], k::self.down_scale[2]]
                    merged_x.append(tmp_x)

        x = torch.cat(merged_x, 1)
        x = self.norm(x)
        x = self.reduction(x)

        return x

class ConvDropoutNormNonlin(nn.Module):
    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv3d, conv_kwargs=None,
                 norm_op=nn.BatchNorm3d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout3d, dropout_op_kwargs=None,
                 nonlin=nn.GELU, nonlin_kwargs=None):
        super(ConvDropoutNormNonlin, self).__init__()
        self.conv = conv_op(input_channels, output_channels, **conv_kwargs)

        if dropout_op is not None and dropout_op_kwargs['p'] is not None and dropout_op_kwargs[
            'p'] > 0:

            self.dropout = dropout_op(**dropout_op_kwargs)
        else:
            self.dropout = None
        self.instnorm = norm_op(output_channels, **norm_op_kwargs)

        self.lrelu = nonlin(**nonlin_kwargs) if nonlin_kwargs != None else nonlin()

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.lrelu(self.instnorm(x))


class DownOrUpSample(nn.Module):
    def __init__(self, input_feature_channels, output_feature_channels,
                 conv_op, conv_kwargs,
                 norm_op=nn.InstanceNorm3d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout3d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, nonlinbasic_block=ConvDropoutNormNonlin):
        super(DownOrUpSample, self).__init__()
        self.blocks = nonlinbasic_block(input_feature_channels, output_feature_channels, conv_op, conv_kwargs,
                                        norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                                        nonlin, nonlin_kwargs)

    def forward(self, x):
        return self.blocks(x)


class DeepSupervision(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.proj = nn.Conv3d(
            dim, num_classes, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        x = self.proj(x)
        return x

class F_BasicLayer(nn.Module):
    def __init__(self, num_stage, num_only_conv_stage, num_pool, base_num_features,
                 norm_op=None,
                 nonlin=None,
                 conv_kernel_sizes=None, conv_pad_sizes=None, pool_op_kernel_sizes=None, max_num_features=None,
                 down_or_upsample=None, feat_map_mul_on_downscale=2,
                 use_checkpoint=False, stack=False):

        super().__init__()

        self.fft_norm = 'ortho'
        self.stack = stack
        self.num_stage = num_stage
        self.num_only_conv_stage = num_only_conv_stage
        self.num_pool = num_pool
        self.use_checkpoint = use_checkpoint
        dim = min((base_num_features * feat_map_mul_on_downscale ** num_stage), max_num_features)
        conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}

        # self.depth = depth
        conv_kwargs['kernel_size'] = conv_kernel_sizes[num_stage]
        conv_kwargs['padding'] = conv_pad_sizes[num_stage]

        self.input_du_channels = dim
        self.output_du_channels = min(int(base_num_features * feat_map_mul_on_downscale ** (num_stage + 1)),
                                      max_num_features)

        self.conv_blocks_out = ConvNormAct(dim, dim, kernel_size=conv_kwargs['kernel_size'], norm=norm_op, act=nonlin)

        self.conv_blocks = F_SimAM_Block(dim * 2, dim * 2, kernel_size=conv_kwargs['kernel_size'], norm=norm_op, act=nonlin)


        # patch merging layer
        if down_or_upsample is not None:
            self.down_or_upsample = PatchMerging(self.input_du_channels, self.output_du_channels, norm=norm_op, proj_type='depthwise', down_scale=pool_op_kernel_sizes[num_stage], kernel_size=[3, 3, 3])
        else:
            self.down_or_upsample = None


    def forward(self, x):
        '''FFT'''
        B, C, D, H, W = x.shape
        fft_dim = (-3, -2, -1)
        ffted = torch.fft.rfftn(x.to(torch.float32), dim=fft_dim, norm=self.fft_norm)

        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)

        clamp = False
        remove = False
        if clamp:
            ffted = torch.clamp(ffted, min=-10, max=10)
        if remove:
            fftedmin10 = torch.clamp(ffted, min=10)
            fftedmax10 = torch.clamp(ffted, max=-10)
            ffted = torch.where(ffted > 0, fftedmax10, fftedmin10)

        ffted = ffted.permute(0, 1, 5, 2, 3, 4).contiguous()  # (batch, c, 2, d, h, w/2+1)
        ffted = ffted.view((B, -1,) + ffted.size()[3:])


        ffted = self.conv_blocks(ffted)

        '''iFFT'''
        ffted = ffted.view((B, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4, 5, 2).contiguous()
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        x = torch.fft.irfftn(ffted, s=x.shape[-3:], dim=fft_dim, norm=self.fft_norm).to(torch.float32)
        x = self.conv_blocks_out(x)

        if self.down_or_upsample is not None:
            du = self.down_or_upsample(x)
        else:
            du = None

        return x, du

class BasicLayer(nn.Module):
    def __init__(self, num_stage, num_only_conv_stage, num_pool, base_num_features, input_resolution, depth, num_heads,
                 window_size,
                 norm_op=None,
                 nonlin=None,
                 conv_kernel_sizes=None, conv_pad_sizes=None, pool_op_kernel_sizes=None, max_num_features=None,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, down_or_upsample=None, feat_map_mul_on_downscale=2,
                 use_checkpoint=False):

        super().__init__()
        self.num_stage = num_stage
        self.num_only_conv_stage = num_only_conv_stage
        self.num_pool = num_pool
        self.use_checkpoint = use_checkpoint
        dim = min((base_num_features * feat_map_mul_on_downscale ** num_stage), max_num_features)
        conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}

        # self.depth = depth
        conv_kwargs['kernel_size'] = conv_kernel_sizes[num_stage]
        conv_kwargs['padding'] = conv_pad_sizes[num_stage]

        self.input_du_channels = dim
        self.output_du_channels = min(int(base_num_features * feat_map_mul_on_downscale ** (num_stage + 1)),
                                      max_num_features)

        self.conv_blocks = SimAM_Block(dim, dim, kernel_size=conv_kwargs['kernel_size'], norm=norm_op, act=nonlin)

        # build blocks
        if num_stage >= num_only_conv_stage:
            self.swin_blocks = nn.ModuleList([
                SwinTransformerBlock(dim=self.input_du_channels, input_resolution=input_resolution,
                                     num_heads=num_heads, window_size=window_size,
                                     shift_size=[0, 0, 0] if (i % 2 == 0) else [
                                         window_size[0] // 2, window_size[1] // 2, window_size[2] // 2],
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias, qk_scale=qk_scale,
                                     drop=drop, attn_drop=attn_drop,
                                     drop_path=drop_path[i] if isinstance(
                                         drop_path, list) else drop_path,
                                     norm_layer=norm_layer)
                for i in range(depth)])

        # patch merging layer
        if down_or_upsample is not None:
            self.down_or_upsample = PatchMerging(self.input_du_channels, self.output_du_channels, norm=norm_op, proj_type='depthwise',
                                                 down_scale=pool_op_kernel_sizes[num_stage], kernel_size=[3, 3, 3])
        else:
            self.down_or_upsample = None


    def forward(self, x):
        B, C, D, H, W = x.shape
        s = x

        x = self.conv_blocks(x)

        if self.num_stage >= self.num_only_conv_stage:
            for tblk in self.swin_blocks:
                if self.use_checkpoint:
                    s = checkpoint.checkpoint(tblk, s)
                else:
                    s = tblk(s)
            x = x + s
        if self.down_or_upsample is not None:
            du = self.down_or_upsample(x)
        else:
            du = None

        return x, du
class BasicLayer_Up(nn.Module):
    def __init__(self, img_size, num_stage, num_only_conv_stage, num_pool, base_num_features, input_resolution, depth, num_heads,
                 window_size,
                 norm_op=None, norm_op_kwargs=None,
                 nonlin=None,
                 conv_kernel_sizes=None, conv_pad_sizes=None, pool_op_kernel_sizes=None, max_num_features=None,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, down_or_upsample=None, feat_map_mul_on_downscale=2,
                 num_classes=None, use_checkpoint=False):

        super().__init__()
        self.img_size = img_size
        self.num_stage = num_stage
        self.num_only_conv_stage = num_only_conv_stage+1
        self.num_pool = num_pool
        self.use_checkpoint = use_checkpoint
        dim = min((base_num_features * feat_map_mul_on_downscale ** num_stage), max_num_features)
        conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}

        if num_stage < num_pool:
            input_features = 2 * dim
        else:
            input_features = dim

        # self.depth = depth
        conv_kwargs['kernel_size'] = conv_kernel_sizes[num_stage]
        conv_kwargs['padding'] = conv_pad_sizes[num_stage]
        dowm_stage = num_stage - 1
        self.up_scale = pool_op_kernel_sizes[dowm_stage]
        self.input_du_channels = dim
        self.output_du_channels = min(int(base_num_features * feat_map_mul_on_downscale ** (num_stage - 1)),
                                      max_num_features)

        self.conv_blocks = SimAM_Block(input_features, dim, kernel_size=conv_kwargs['kernel_size'], norm=norm_op, act=nonlin)
        self.reduction = nn.Conv3d(input_features, dim, kernel_size=1)
        self.out = outconv(dim, num_classes, kernel_size=conv_kwargs['kernel_size'], norm=norm_op, act=nonlin)
        # build blocks
        if num_stage >= self.num_only_conv_stage:
            self.swin_blocks = nn.ModuleList([
                SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                     num_heads=num_heads, window_size=window_size,
                                     shift_size=[0, 0, 0] if (i % 2 == 0) else [
                                         window_size[0] // 2, window_size[1] // 2, window_size[2] // 2],
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias, qk_scale=qk_scale,
                                     drop=drop, attn_drop=attn_drop,
                                     drop_path=drop_path[i] if isinstance(
                                         drop_path, list) else drop_path,
                                     norm_layer=norm_layer)
                for i in range(depth)])

        # patch merging layer
        if down_or_upsample is not None:
            self.down_or_upsample = nn.Sequential(nn.Conv3d(self.input_du_channels, self.output_du_channels, 1,
                                                                   1, bias=False),
                                                  norm_op(self.output_du_channels, **norm_op_kwargs))
        else:
            self.down_or_upsample = None

        self.deep_supervision = DeepSupervision(dim, num_classes)


    def forward(self, x, skip):
        s = x
        if self.num_stage < self.num_pool:
            x = torch.cat((x, skip), dim=1)

        x = self.conv_blocks(x)

        if self.num_stage >= self.num_only_conv_stage:
            if self.num_stage < self.num_pool:
                s = s + skip
            for tblk in self.swin_blocks:
                if self.use_checkpoint:
                    s = checkpoint.checkpoint(tblk, s)
                else:
                    s = tblk(s)
            x = x + s

        if self.down_or_upsample is not None:
            du = F.interpolate(x, scale_factor=self.up_scale, mode='trilinear', align_corners=True)
            du = self.down_or_upsample(du)

        ds = self.deep_supervision(x)

        if self.down_or_upsample is not None:
            return du, ds
        elif self.down_or_upsample is None:
            x = self.out(x)
            return x, ds


class PSFH_Net(nn.Module):
    def __init__(self, img_size, base_num_features, num_classes, image_channels=1, num_only_conv_stage=2,
                 feat_map_mul_on_downscale=2, pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None, deep_supervision=True, max_num_features=None, depths=None, num_heads=None,
                 window_size=None, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., dropout_p=0.1, drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm, use_checkpoint=False, positional_encoding=True, stack=True):
        super().__init__()
        self.num_classes = num_classes
        self.conv_op = nn.Conv3d
        norm_op = nn.BatchNorm3d
        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op = nn.Dropout3d
        dropout_op_kwargs = {'p': dropout_p, 'inplace': True}
        nonlin = nn.GELU

        self.do_ds = deep_supervision
        self.num_pool = len(pool_op_kernel_sizes)
        conv_pad_sizes = []
        for krnl in conv_kernel_sizes:
            conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])
        dpr = [x.item() for x in torch.linspace(
            0, drop_path_rate, sum(depths))]
        # build layers
        conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}
        conv_kwargs['kernel_size'] = conv_kernel_sizes[0]
        conv_kwargs['padding'] = conv_pad_sizes[0]
        '''share stem'''
        self.inconv = inconv(image_channels, base_num_features, norm=norm_op, act=nonlin, kernel_size=[3,3,3])
        '''stem patchmering'''
        self.patchmerging = PatchMerging(base_num_features, base_num_features, norm=norm_op, proj_type='depthwise', down_scale=[1, 2, 2],
                                         kernel_size=[3, 3, 3])

        self.map_fusion = SemanticMapFusion(in_dim_list=[64, 128, 256], dim=320, heads=4, depth=2,
                                            norm=norm_op, ps=positional_encoding)


        self.reduction_out = nn.Conv3d(320 * 2, 320, kernel_size=1, bias=False)

        self.f_down_layers = nn.ModuleList()
        for i_layer in range(self.num_pool):  # 0,1,2,3
            layer = F_BasicLayer(num_stage=i_layer, num_only_conv_stage=num_only_conv_stage, num_pool=self.num_pool,
                               base_num_features=base_num_features,
                               norm_op=norm_op,
                               nonlin=nonlin,
                               conv_kernel_sizes=conv_kernel_sizes, conv_pad_sizes=conv_pad_sizes, pool_op_kernel_sizes=pool_op_kernel_sizes,
                               max_num_features=max_num_features,
                               down_or_upsample=True,
                               feat_map_mul_on_downscale=feat_map_mul_on_downscale,
                               use_checkpoint=use_checkpoint,
                               stack=stack
                               )
            self.f_down_layers.append(layer)



        self.down_layers = nn.ModuleList()
        for i_layer in range(self.num_pool):  # 0,1,2,3
            layer = BasicLayer(num_stage=i_layer, num_only_conv_stage=num_only_conv_stage, num_pool=self.num_pool,
                               base_num_features=base_num_features,
                               input_resolution=(
                                       (64,64,64) // np.prod(pool_op_kernel_sizes[:i_layer], 0, dtype=np.int64)),  # 56,28,14,7
                               depth=depths[i_layer - num_only_conv_stage] if (
                                       i_layer >= num_only_conv_stage) else None,
                               num_heads=num_heads[i_layer - num_only_conv_stage] if (
                                       i_layer >= num_only_conv_stage) else None,
                               window_size=window_size,
                               norm_op=norm_op,
                               nonlin=nonlin,
                               conv_kernel_sizes=conv_kernel_sizes, conv_pad_sizes=conv_pad_sizes, pool_op_kernel_sizes=pool_op_kernel_sizes,
                               max_num_features=max_num_features,
                               mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer - num_only_conv_stage]):sum(depths[:i_layer - num_only_conv_stage + 1])] if (
                                       i_layer >= num_only_conv_stage) else None,
                               norm_layer=norm_layer,
                               down_or_upsample=nn.Conv3d,
                               feat_map_mul_on_downscale=feat_map_mul_on_downscale,
                               use_checkpoint=use_checkpoint
                               )
            self.down_layers.append(layer)
        self.up_layers = nn.ModuleList()
        for i_layer in range(self.num_pool + 1)[::-1]:
            layer = BasicLayer_Up(img_size=img_size, num_stage=i_layer, num_only_conv_stage=num_only_conv_stage, num_pool=self.num_pool,
                               base_num_features=base_num_features,
                               input_resolution=(
                                       (64,64,64) // np.prod(pool_op_kernel_sizes[:i_layer], 0, dtype=np.int64)),
                               depth=depths[i_layer - num_only_conv_stage] if (
                                       i_layer >= num_only_conv_stage) else None,
                               num_heads=num_heads[i_layer - num_only_conv_stage] if (
                                       i_layer >= num_only_conv_stage) else None,
                               window_size=window_size,
                               norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                               nonlin=nonlin,
                               conv_kernel_sizes=conv_kernel_sizes, conv_pad_sizes=conv_pad_sizes, pool_op_kernel_sizes=pool_op_kernel_sizes,
                               max_num_features=max_num_features,
                               mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer - num_only_conv_stage]):sum(depths[:i_layer - num_only_conv_stage + 1])] if (
                                       i_layer >= num_only_conv_stage) else None,
                               norm_layer=norm_layer,
                               down_or_upsample = True if (
                                       i_layer > 0) else None,
                               feat_map_mul_on_downscale=feat_map_mul_on_downscale,
                               num_classes=self.num_classes,
                               use_checkpoint=use_checkpoint,
                                  )
            self.up_layers.append(layer)

        self.apply(self._InitWeights)




    def _InitWeights(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=.02)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, x):
        """stem"""
        B, C, D, H, W = x.shape
        x = self.inconv(x)
        x = self.patchmerging(x)
        '''Frequency domain conv'''
        f = x
        sf_list = []
        for idx, layer in enumerate(self.f_down_layers):
            sf, f = layer(f)
            sf_list.append(sf)
        '''Spatial domain'''
        x_skip = []
        for idx, layer in enumerate(self.down_layers):
            s, x = layer(x)
            '''layer add'''
            x_skip.append(s)
        out = []

        '''concat freq & spatial'''
        x = self.reduction_out(torch.concat([x, f], dim=1))

        x_skip = self.map_fusion(x_skip, sf_list)
        # x_skip = self.map_fusion(x_skip)

        for inx, layer in enumerate(self.up_layers):
            x, ds = layer(x, x_skip[self.num_pool - inx]) if inx > 0 else layer(x, None)

            if inx > 1:
                out.append(ds)

        if self.do_ds:
            for i in range(len(out)):
                out[i] = F.interpolate(out[i], size=(D, H, W), mode='trilinear', align_corners=True)
            out.append(x)
            return out[::-1]
        else:
            out.append(x)
            return out[-1]

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}