import numpy as np
import torch
import torch.nn as nn
import pdb


def get_model(args, pretrain=False):
        model = PSFH_Net(img_size=args.training_size, base_num_features=args.base_chan, num_classes=args.classes,
                        image_channels=1, num_only_conv_stage=args.num_only_conv_stage,
                        num_conv_per_stage=args.num_conv_per_stage,
             feat_map_mul_on_downscale=2, pool_op_kernel_sizes=args.pool_op_kernel_sizes,
             conv_kernel_sizes=args.conv_kernel_sizes, deep_supervision=True, max_num_features=args.max_num_features, depths=args.depths,
                        num_heads=args.num_heads,
             window_size=args.window_sizes, mlp_ratio=args.mlp_ratio, qkv_bias=True, qk_scale=None,
             drop_rate=0., attn_drop_rate=0., dropout_p=0.1, drop_path_rate=0.2,
             norm_layer=nn.LayerNorm, use_checkpoint=False, positional_encoding=args.positional_encoding, stack=args.stack)

        return model