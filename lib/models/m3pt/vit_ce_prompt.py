import math
import logging
import pdb
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import to_2tuple

from lib.models.layers.patch_embed import PatchEmbed
from .utils import combine_tokens, recover_tokens, token2feature, feature2token
from .vit import VisionTransformer
from ..layers.attn_blocks import CEBlock, candidate_elimination_prompt

_logger = logging.getLogger(__name__)


class Fovea(nn.Module):

    def __init__(self, smooth=False):
        super().__init__()

        self.softmax = nn.Softmax(dim=-1)

        self.smooth = smooth
        if smooth:
            self.smooth = nn.Parameter(torch.zeros(1) + 10.0)


    def forward(self, x):
        '''
            x: [batch_size, features, k]
        '''
        b, c, h, w = x.shape

        x = x.contiguous().view(b, c, h*w)
        if self.smooth:
            mask = self.softmax(x * self.smooth)
        else:
            mask = self.softmax(x)
        # output = mask * x
        # output = output.contiguous().view(b, c, h, w)

        return mask.contiguous().view(b, c, h, w)


class Prompt_block(nn.Module, ):
    def __init__(self, inplanes=None, hide_channel=None, smooth=False):  # inplanes为嵌入维度，hide_channel为8
        super(Prompt_block, self).__init__()
        self.conv0_0 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv0_1 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv1x1 = nn.Conv2d(in_channels=hide_channel, out_channels=inplanes, kernel_size=1, stride=1, padding=0)
        self.fovea = Fovea(smooth=smooth)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """ Forward pass with input x. """
        B, C, W, H = x.shape
        # x0: RGB模态
        x0 = x[:, 0:int(C/2), :, :].contiguous()
        x0 = self.conv0_0(x0)
        x1 = x[:, int(C/2):, :, :].contiguous()
        x1 = self.conv0_1(x1)
        x0 = self.fovea(x0) * x0 + x1

        return self.conv1x1(x0)


class Prompt_MA_Block(nn.Module):
    def __init__(self, inplanes=None, hide_channel=None, drop=None):
        super(Prompt_MA_Block, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv1_0 = nn.Conv2d(in_channels=hide_channel, out_channels=hide_channel, kernel_size=1, stride=1,
                                 padding=0, groups=hide_channel)
        self.conv1_1 = nn.Conv2d(in_channels=hide_channel, out_channels=hide_channel, kernel_size=1, stride=1,
                                 padding=0)
        self.conv1_2 = nn.Conv2d(in_channels=hide_channel, out_channels=hide_channel, kernel_size=1, stride=1,
                                 dilation=2, padding=0)
        self.dropout = nn.Dropout(drop)
        # self.act_layer = Fovea(smooth=False)
        self.act_layer = nn.Sequential(nn.BatchNorm2d(hide_channel), nn.GELU())  # nn.ReLU(inplace=True))
        self.conv2 = nn.Conv2d(in_channels=hide_channel, out_channels=inplanes, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        x_ori = x
        x = self.conv0(x)
        x = x_ori + self.conv2(self.act_layer(self.dropout(self.conv1_1(x) + self.conv1_0(x) + self.conv1_2(x))))
        return F.relu(x)


class Prompt_Fusion_Block(nn.Module):
    def __init__(self, inplanes=None, hide_channel=None, drop=None):
        super(Prompt_Fusion_Block, self).__init__()
        self.hide_channel = hide_channel
        self.conv_low_rgb = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv_low_dte = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv_weight = nn.Conv2d(in_channels=5 * hide_channel, out_channels=2 * hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Conv2d(in_channels=hide_channel, out_channels=inplanes, kernel_size=1, stride=1, padding=0)
        self.softmax = Fovea(smooth=False)
        self.dropout = nn.Dropout(drop)

    def forward(self, f_rgb, f_dte):

        f_rgb = self.conv_low_rgb(f_rgb)
        f_dte = self.conv_low_dte(f_dte)

        f_total = f_rgb + f_dte
        f_share = f_rgb * f_dte
        f_rgb_spec = f_rgb - f_dte
        f_dte_spec = f_dte - f_rgb

        f = torch.cat([f_rgb_spec, f_share, f_total, f_share, f_dte_spec], dim=1)
        weight = self.softmax(self.dropout(self.conv_weight(f)))
        f_fusion = f_share + f_rgb * weight[:, :self.hide_channel] + f_dte * weight[:, self.hide_channel:]
        return self.conv(f_fusion)


class Prompt_SceneAdapter(nn.Module):
    def __init__(self, inplanes=None, hide_dim=None, drop=None):
        super(Prompt_SceneAdapter, self).__init__()
        self.conv0_0 = nn.Conv2d(in_channels=inplanes, out_channels=hide_dim, kernel_size=1, stride=1, padding=0)
        self.conv0_1 = nn.Conv2d(in_channels=inplanes, out_channels=hide_dim, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(in_channels=hide_dim, out_channels=inplanes, kernel_size=1, stride=1, padding=0)
        self.spatial_gate_rgb = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True))
        self.spatial_gate_dte = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True))
        self.conv_scene = nn.Sequential(
            nn.Conv2d(in_channels=hide_dim, out_channels=2, kernel_size=1, stride=1, padding=0))
        self.conv_adapt_rgb = nn.Sequential(
            nn.Conv2d(in_channels=hide_dim + 2, out_channels=hide_dim, kernel_size=1, stride=1, padding=0))
        self.conv_adapt_dte = nn.Sequential(
            nn.Conv2d(in_channels=hide_dim + 2, out_channels=hide_dim, kernel_size=1, stride=1, padding=0))
        self.drop = nn.Dropout(drop)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def channel_pool(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

    def forward(self, feature_rgb, feature_dte):
        low_rgb = self.conv0_0(feature_rgb)
        low_dte = self.conv0_1(feature_dte)
        rgb_spatial_attention = torch.sigmoid(self.spatial_gate_rgb(self.channel_pool(low_rgb)))
        dte_spatial_attention = torch.sigmoid(self.spatial_gate_dte(self.channel_pool(low_dte)))
        credit_map_rgb = self.conv_scene(low_rgb * rgb_spatial_attention)
        credit_map_dte = self.conv_scene(low_dte * dte_spatial_attention)

        prompted_rgb = self.drop(self.conv_adapt_rgb(torch.cat([low_rgb, credit_map_dte], dim=1)))
        prompted_dte = self.drop(self.conv_adapt_dte(torch.cat([low_dte, credit_map_rgb], dim=1)))

        feature_rgb = feature_rgb + self.conv1(prompted_rgb)
        feature_dte = feature_dte + self.conv1(prompted_dte)
        return feature_rgb, feature_dte


class Prompt_InterAdapter(nn.Module):
    def __init__(self, inplanes=None, hide_dim=None, drop=None):
        super(Prompt_InterAdapter, self).__init__()
        self.low_conv = nn.Conv2d(in_channels=inplanes, out_channels=hide_dim, kernel_size=1, stride=1, padding=0)
        self.low_conv_prompt = nn.Conv2d(in_channels=inplanes, out_channels=hide_dim, kernel_size=1, stride=1, padding=0)
        self.high_conv = nn.Conv2d(in_channels=hide_dim, out_channels=inplanes, kernel_size=1, stride=1, padding=0)
        self.adapt_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(2, eps=1e-5, momentum=0.01, affine=True))
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels=hide_dim+1, out_channels=hide_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(hide_dim, eps=1e-5, momentum=0.01, affine=True))
        self.drop = nn.Dropout(drop)

    def forward(self, feature_prompt, feature):
        low_feature_prompt = self.low_conv_prompt(feature_prompt)
        low_feature = self.low_conv(feature)
        # inter_prompt = self.drop(self.adapt_conv(torch.mean(low_feature_prompt, 1).unsqueeze(1)))
        inter_prompt = torch.mean(low_feature_prompt, 1).unsqueeze(1)
        prompted_feature = self.drop(self.fusion_conv(torch.cat([low_feature, inter_prompt], dim=1)))
        prompted_feature = self.high_conv(prompted_feature)
        return prompted_feature + feature



class VisionTransformerCE(VisionTransformer):
    """ Vision Transformer with candidate elimination (CE) module

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', ce_loc=None, ce_keep_ratio=None, search_size=None, template_size=None,
                 new_patch_size=None, prompt_type=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
            new_patch_size: backbone stride
        """
        super().__init__()
        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = to_2tuple(img_size)
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim)) # it's redundant
        self.pos_drop = nn.Dropout(p=drop_rate)

        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_search=new_P_H * new_P_W
        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_template = new_P_H * new_P_W

        """add here, no need use backbone.finetune_track """
        self.pos_embed_z = nn.Parameter(torch.zeros(1, self.num_patches_template, embed_dim))
        self.pos_embed_x = nn.Parameter(torch.zeros(1, self.num_patches_search, embed_dim))

        self.prompt_type = prompt_type

        self.ma_loc = [2, 5, 8]
        self.fusion_loc = 10
        self.prompt_token_num = 2

        if self.prompt_type in ['vipt_shaw', 'm3pt']:
            self.drop = 0.1
            prompt_blocks = []
            block_nums = depth if self.prompt_type == 'm3pt' else 1
            for i in range(block_nums - self.fusion_loc):
                prompt_blocks.append(Prompt_block(inplanes=embed_dim, hide_channel=8, smooth=True))
            self.prompt_blocks = nn.Sequential(*prompt_blocks)

            prompt_norms = []
            for i in range(block_nums):
                prompt_norms.append(norm_layer(embed_dim))
            self.prompt_norms = nn.Sequential(*prompt_norms)

            prompt_ma_rgb_blocks = []
            prompt_ma_dte_blocks = []
            for i in range(len(self.ma_loc)):
                prompt_ma_rgb_blocks.append(Prompt_MA_Block(inplanes=embed_dim, hide_channel=8, drop=self.drop))
                prompt_ma_dte_blocks.append(Prompt_MA_Block(inplanes=embed_dim, hide_channel=8, drop=self.drop))
            self.prompt_ma_rgb_blocks = nn.Sequential(*prompt_ma_rgb_blocks)
            self.prompt_ma_dte_blocks = nn.Sequential(*prompt_ma_dte_blocks)

            prompt_sceneadapter_blocks = []
            for i in range(self.fusion_loc):
                prompt_sceneadapter_blocks.append(Prompt_InterAdapter(inplanes=embed_dim, hide_dim=8, drop=self.drop))
            self.prompt_sceneadapter_blocks = nn.Sequential(*prompt_sceneadapter_blocks)
            self.prompt_fusion_blocks = Prompt_Fusion_Block(inplanes=embed_dim, hide_channel=16, drop=self.drop)

            self.prompt_rgb_tokens = nn.Parameter(torch.zeros(1, self.prompt_token_num, embed_dim))
            self.prompt_dte_tokens = nn.Parameter(torch.zeros(1, self.prompt_token_num, embed_dim))
            self.prompt_fusion_tokens = nn.Parameter(torch.zeros(1, self.prompt_token_num, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        blocks = []
        ce_index = 0
        self.ce_loc = ce_loc
        for i in range(depth):
            ce_keep_ratio_i = 1.0
            if ce_loc is not None and i in ce_loc:
                ce_keep_ratio_i = ce_keep_ratio[ce_index]
                ce_index += 1
            blocks.append(
                CEBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                    keep_ratio_search=ce_keep_ratio_i)
            )

        self.blocks = nn.Sequential(*blocks)
        self.norm = norm_layer(embed_dim)

        self.init_weights(weight_init)

    def forward_features(self, z, x, mask_z=None, mask_x=None,
                         ce_template_mask=None, ce_keep_rate=None,
                         return_last_attn=False):

        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        # rgb_img
        x_rgb = x[:, :3, :, :]
        z_rgb = z[:, :3, :, :]

        x_dte = x[:, 3:, :, :]
        z_dte = z[:, 3:, :, :]

        z_rgb = self.patch_embed(z_rgb)
        x_rgb = self.patch_embed(x_rgb)
        z_dte = self.patch_embed(z_dte)
        x_dte = self.patch_embed(x_dte)

        # attention mask handling
        # B, H, W
        if mask_z is not None and mask_x is not None:
            mask_z = F.interpolate(mask_z[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_z = mask_z.flatten(1).unsqueeze(-1)

            mask_x = F.interpolate(mask_x[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_x = mask_x.flatten(1).unsqueeze(-1)

            mask_x = combine_tokens(mask_z, mask_x, mode=self.cat_mode)
            mask_x = mask_x.squeeze(-1)

        z_rgb += self.pos_embed_z
        x_rgb += self.pos_embed_x
        z_dte += self.pos_embed_z
        x_dte += self.pos_embed_x

        x_rgb = combine_tokens(z_rgb, x_rgb, mode=self.cat_mode)
        x_dte = combine_tokens(z_dte, x_dte, mode=self.cat_mode)

        x_rgb = self.pos_drop(x_rgb)
        x_dte = self.pos_drop(x_dte)

        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]

        global_index_t_rgb = torch.linspace(0, lens_z - 1, lens_z, dtype=torch.int64).to(x_rgb.device)
        global_index_t_rgb = global_index_t_rgb.repeat(B, 1)

        global_index_s_rgb = torch.linspace(0, lens_x - 1, lens_x, dtype=torch.int64).to(x_rgb.device)
        global_index_s_rgb = global_index_s_rgb.repeat(B, 1)

        global_index_t_dte = torch.linspace(0, lens_z - 1, lens_z, dtype=torch.int64).to(x_dte.device)
        global_index_t_dte = global_index_t_dte.repeat(B, 1)
        global_index_s_dte = torch.linspace(0, lens_x - 1, lens_x, dtype=torch.int64).to(x_dte.device)
        global_index_s_dte = global_index_s_dte.repeat(B, 1)

        removed_indexes_s = []
        removed_flag = False

        prompt_rgb_token = self.prompt_rgb_tokens
        prompt_dte_token = self.prompt_dte_tokens
        # print(prompt_token.shape)
        x_rgb = combine_tokens(x_rgb, prompt_rgb_token.expand(B, -1, -1), mode=self.cat_mode)
        x_dte = combine_tokens(x_dte, prompt_dte_token.expand(B, -1, -1), mode=self.cat_mode)

        for i, blk in enumerate(self.blocks):
            if i < self.fusion_loc:
                if i in self.ma_loc:
                    z_rgb_ori = x_rgb[:, :lens_z]
                    x_rgb_ori = x_rgb[:, lens_z:-self.prompt_token_num]
                    prompt_rgb_token = x_rgb[:, -self.prompt_token_num:]

                    z_rgb_ori = self.prompt_norms[i](z_rgb_ori)
                    x_rgb_ori = self.prompt_norms[i](x_rgb_ori)
                    prompt_rgb_token = self.prompt_norms[i](prompt_rgb_token)

                    z_rgb_feat = token2feature(z_rgb_ori)
                    x_rgb_feat = token2feature(x_rgb_ori)
                    x_rgb_feat = self.prompt_ma_rgb_blocks[i//3](x_rgb_feat)
                    z_rgb_feat = self.prompt_ma_rgb_blocks[i//3](z_rgb_feat)

                    z_rgb_ori = feature2token(z_rgb_feat)
                    x_rgb_ori = feature2token(x_rgb_feat)
                    x_rgb_ori = torch.cat([z_rgb_ori, x_rgb_ori, prompt_rgb_token], dim=1)

                    x_rgb, global_index_t_rgb, global_index_s_rgb, removed_index_s_rgb, attn_rgb = \
                        blk(x_rgb, global_index_t_rgb, global_index_s_rgb, mask_x, ce_template_mask, ce_keep_rate)
                    x_rgb += x_rgb_ori

                    z_rgb = x_rgb[:, :lens_z]
                    z_rgb_feat = token2feature(z_rgb)
                    z_dte = x_dte[:, :lens_z]
                    z_dte_feat = token2feature(z_dte)
                    z_dte_feat = self.prompt_sceneadapter_blocks[i](z_rgb_feat, z_dte_feat)
                    z_dte = feature2token(z_dte_feat)
                    x_dte = torch.cat([z_dte, x_dte[:, lens_z:]], dim=1)


                    z_dte_ori = x_dte[:, :lens_z]
                    x_dte_ori = x_dte[:, lens_z:-self.prompt_token_num]
                    prompt_dte_token = x_dte[:, -self.prompt_token_num:]
                    z_dte_ori = self.prompt_norms[i](z_dte_ori)
                    x_dte_ori = self.prompt_norms[i](x_dte_ori)
                    prompt_dte_token = self.prompt_norms[i](prompt_dte_token)  ###
                    z_dte_feat = token2feature(z_dte_ori)
                    x_dte_feat = token2feature(x_dte_ori)
                    x_dte_feat = self.prompt_ma_dte_blocks[i//3](x_dte_feat)
                    z_dte_feat = self.prompt_ma_dte_blocks[i//3](z_dte_feat)
                    z_dte_ori = feature2token(z_dte_feat)
                    x_dte_ori = feature2token(x_dte_feat)
                    x_dte_ori = torch.cat([z_dte_ori, x_dte_ori, prompt_dte_token], dim=1)

                    x_dte, global_index_t_dte, global_index_s_dte, removed_index_s_dte, attn_dte = \
                        blk(x_dte, global_index_t_dte, global_index_s_dte, mask_x, ce_template_mask, ce_keep_rate)
                    x_dte += x_dte_ori

                    z_rgb = x_rgb[:, :lens_z]
                    z_rgb_feat = token2feature(z_rgb)
                    z_dte = x_dte[:, :lens_z]
                    z_dte_feat = token2feature(z_dte)
                    z_rgb_feat = self.prompt_sceneadapter_blocks[i](z_dte_feat, z_rgb_feat)
                    z_rgb = feature2token(z_rgb_feat)
                    x_rgb = torch.cat([z_rgb, x_rgb[:, lens_z:]], dim=1)

                else:
                    x_rgb, global_index_t_rgb, global_index_s_rgb, removed_index_s_rgb, attn_rgb = \
                        blk(x_rgb, global_index_t_rgb, global_index_s_rgb, mask_x, ce_template_mask, ce_keep_rate)

                    z_rgb = x_rgb[:, :lens_z]
                    z_rgb_feat = token2feature(z_rgb)
                    z_dte = x_dte[:, :lens_z]
                    z_dte_feat = token2feature(z_dte)
                    z_dte_feat = self.prompt_sceneadapter_blocks[i](z_rgb_feat, z_dte_feat)
                    z_dte = feature2token(z_dte_feat)
                    x_dte = torch.cat([z_dte, x_dte[:, lens_z:]], dim=1)

                    x_dte, global_index_t_dte, global_index_s_dte, removed_index_s_dte, attn_dte = \
                        blk(x_dte, global_index_t_dte, global_index_s_dte, mask_x, ce_template_mask, ce_keep_rate)

                    z_rgb = x_rgb[:, :lens_z]
                    z_rgb_feat = token2feature(z_rgb)
                    z_dte = x_dte[:, :lens_z]
                    z_dte_feat = token2feature(z_dte)
                    z_rgb_feat = self.prompt_sceneadapter_blocks[i](z_dte_feat, z_rgb_feat)
                    z_rgb = feature2token(z_rgb_feat)
                    x_rgb = torch.cat([z_rgb, x_rgb[:, lens_z:]], dim=1)


            elif i == self.fusion_loc:
                prompt_rgb_token = x_rgb[:, -self.prompt_token_num:]
                z_rgb = x_rgb[:, :lens_z]
                x_rgb = x_rgb[:, lens_z:-self.prompt_token_num]

                prompt_dte_token = x_dte[:, -self.prompt_token_num:]
                z_dte = x_dte[:, :lens_z]
                x_dte = x_dte[:, lens_z:-self.prompt_token_num]

                prompt_fusion_token = self.prompt_norms[i](prompt_rgb_token + prompt_dte_token)
                new_prompt_fusion_token = self.prompt_fusion_tokens
                prompt_fusion_token += new_prompt_fusion_token.expand(B, -1, -1)

                z_rgb = self.prompt_norms[i](z_rgb)
                x_rgb = self.prompt_norms[i](x_rgb)
                z_dte = self.prompt_norms[i](z_dte)
                x_dte = self.prompt_norms[i](x_dte)
                z_rgb_feat = token2feature(z_rgb)
                x_rgb_feat = token2feature(x_rgb)
                z_dte_feat = token2feature(z_dte)
                x_dte_feat = token2feature(x_dte)
                x_feat = self.prompt_fusion_blocks(x_rgb_feat, x_dte_feat)
                z_feat = self.prompt_fusion_blocks(z_rgb_feat, z_dte_feat)
                x_prompted = feature2token(x_feat)
                z_prompted = feature2token(z_feat)
                x_prompted = self.prompt_norms[i](x_prompted)
                z_prompted = self.prompt_norms[i](z_prompted)

                x = torch.cat([z_rgb+z_prompted, x_rgb+x_prompted, prompt_fusion_token], dim=1)
                global_index_t = global_index_t_rgb
                global_index_s = global_index_s_rgb
                x, global_index_t, global_index_s, removed_index_s, attn = \
                    blk(x, global_index_t, global_index_s, mask_x, ce_template_mask, ce_keep_rate)
                if self.ce_loc is not None and i in self.ce_loc:
                    removed_indexes_s.append(removed_index_s)

            else:
                x_ori = x
                lens_z_new = global_index_t.shape[1]
                lens_x_new = global_index_s.shape[1]
                prompt_fusion_token = x[:, -self.prompt_token_num:]              ###
                prompt_fusion_token = self.prompt_norms[i](prompt_fusion_token)  ###
                z = x[:, :lens_z_new]
                x = x[:, lens_z_new:-self.prompt_token_num]

                if removed_indexes_s and removed_indexes_s[0] is not None:
                    removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)
                    pruned_lens_x = lens_x - lens_x_new
                    pad_x = torch.zeros([B, pruned_lens_x, x.shape[2]], device=x.device)
                    x = torch.cat([x, pad_x], dim=1)
                    index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)
                    C = x.shape[-1]
                    x = torch.zeros_like(x).scatter_(dim=1,
                                                     index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64),
                                                     src=x)
                x = recover_tokens(x, lens_z_new, lens_x, mode=self.cat_mode)
                x = torch.cat([z, x], dim=1)

                x = self.prompt_norms[i](x)  # todo
                z_tokens = x[:, :lens_z, :]
                x_tokens = x[:, lens_z:, :]
                z_feat = token2feature(z_tokens)
                x_feat = token2feature(x_tokens)

                z_prompted = self.prompt_norms[i](z_prompted)
                x_prompted = self.prompt_norms[i](x_prompted)
                z_prompt_feat = token2feature(z_prompted)
                x_prompt_feat = token2feature(x_prompted)

                z_feat = torch.cat([z_feat, z_prompt_feat], dim=1)
                x_feat = torch.cat([x_feat, x_prompt_feat], dim=1)
                z_feat = self.prompt_blocks[i - self.fusion_loc - 1](z_feat)
                x_feat = self.prompt_blocks[i - self.fusion_loc - 1](x_feat)

                z = feature2token(z_feat)
                x = feature2token(x_feat)
                z_prompted, x_prompted = z, x

                x = combine_tokens(z, x, mode=self.cat_mode)
                ###
                x = x_ori[:, :-self.prompt_token_num] + candidate_elimination_prompt(x, global_index_t.shape[1], global_index_s)
                x = combine_tokens(x, prompt_fusion_token, mode=self.cat_mode)

                x, global_index_t, global_index_s, removed_index_s, attn = \
                    blk(x, global_index_t, global_index_s, mask_x, ce_template_mask, ce_keep_rate)
                x[:, -self.prompt_token_num:] += prompt_fusion_token
                if self.ce_loc is not None and i in self.ce_loc:
                    removed_indexes_s.append(removed_index_s)

        x = self.norm(x)
        lens_x_new = global_index_s.shape[1]
        lens_z_new = global_index_t.shape[1]

        z = x[:, :lens_z_new]
        x = x[:, lens_z_new:-self.prompt_token_num]

        if removed_indexes_s and removed_indexes_s[0] is not None:
            removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)

            pruned_lens_x = lens_x - lens_x_new
            pad_x = torch.zeros([B, pruned_lens_x, x.shape[2]], device=x.device)
            x = torch.cat([x, pad_x], dim=1)
            index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)
            # recover original token order
            C = x.shape[-1]
            x = torch.zeros_like(x).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=x)

        x = recover_tokens(x, lens_z_new, lens_x, mode=self.cat_mode)

        # re-concatenate with the template, which may be further used by other modules
        x = torch.cat([z, x], dim=1)

        aux_dict = {
            "attn": attn,
            "removed_indexes_s": removed_indexes_s,  # used for visualization
        }

        return x, aux_dict

    def forward(self, z, x, ce_template_mask=None, ce_keep_rate=None,
                tnc_keep_rate=None,
                return_last_attn=False):

        x, aux_dict = self.forward_features(z, x, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate,)

        return x, aux_dict


def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformerCE(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")

            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
            print('Load pretrained OSTrack from: ' + pretrained)
            print(f"missing_keys: {missing_keys}")
            print(f"unexpected_keys: {unexpected_keys}")

    return model


def vit_base_patch16_224_ce_prompt(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model


def vit_large_patch16_224_ce_prompt(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model
