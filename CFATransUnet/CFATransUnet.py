# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from functools import partial
import copy
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import _cfg
import math
from .CCFT import CCFT
from .CCFA import CCFA


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        assert max(patch_size) > stride, "Set larger patch_size than stride"

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class OverlapPatchEmbed_up(nn.Module):

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        assert max(patch_size) > stride, "Set larger patch_size than stride"

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] * stride, img_size[1] * stride
        self.num_patches = self.H * self.W

        self.proj_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_chans,embed_dim, kernel_size=1, stride=1),
        )

        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj_1(x)
        B, C, H, W = x.shape

        return x, H, W

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class PPM(nn.Module):
    def __init__(self, pooling_sizes=(1, 3, 5)):
        super().__init__()
        self.layer = nn.ModuleList([nn.AdaptiveAvgPool2d(output_size=(size, size)) for size in pooling_sizes])

    def forward(self, feat):
        b, c, h, w = feat.shape
        output = [layer(feat).view(b, c, -1) for layer in self.layer]
        output = torch.cat(output, dim=-1)
        return output

class ESA_qkv(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, kernel_size=1, stride=1, padding=0, bias=False)
        self.ppm = PPM(pooling_sizes=(1, 3, 5))

    def forward(self, x):
        # input x (b, c, h, w)
        b, c, h, w = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=1)  # q/k/v shape: (b, inner_dim, h, w)
        q = rearrange(q, 'b (head d) h w -> b head (h w) d', head=self.heads)  # q shape: (b, head, n_q, d)

        k, v = self.ppm(k), self.ppm(v)  # k/v shape: (b, inner_dim, n_kv)

        return q,k,v

# Efficient self attention

class ESA_layer(nn.Module):
    def __init__(self, dim,heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q,k,v):

        k = rearrange(k, 'b (head d) n -> b head n d', head=self.heads)  # k shape: (b, head, n_kv, d)
        v = rearrange(v, 'b (head d) n -> b head n d', head=self.heads)  # v shape: (b, head, n_kv, d)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # shape: (b, head, n_q, n_kv)
        attn = self.attend(dots)
        out = torch.matmul(attn, v)  # shape: (b, head, n_q, d)
        out = rearrange(out, 'b head n d -> b n (head d)')
        return self.to_out(out)

class Cat_kv(nn.Module):
    def __init__(self,dim):
        super(Cat_kv, self).__init__()
        self.num = len(dim)
        self.dim = dim
        self.dim_cat = [sum(self.dim[0::]),sum(self.dim[1::]),sum(self.dim[2::]),sum(self.dim[3::])]
        self.dim_change = nn.ModuleList()
        for i_layter in range(self.num):
            layter = nn.Sequential(
                    # nn.Conv2d(self.dim_cat[i_layter],self.dim[i_layter],kernel_size=1,stride=1,padding=0),
                    nn.Conv2d(self.dim[i_layter],self.dim[i_layter],kernel_size=1,stride=1,padding=0),
                    nn.BatchNorm2d(self.dim[i_layter]),
                    nn.ReLU(inplace=True),)
            self.dim_change.append(layter)


    def forward(self,feature):
        list_cat = []
        num = len(feature)
        for i in range(num):
            # feature[i] = torch.cat(feature[i::],dim=1)
            feature[i] = sum(feature[i::])
            feature[i] = self.dim_change[i](feature[i].unsqueeze(-1))
            list_cat.append(feature[i].squeeze(-1))

        return list_cat



class ESA_blcok(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, mlp_dim=512, dropout=0.):
        super().__init__()
        self.num = len(dim)
        self.dim = dim
        self.ESAqkv = nn.ModuleList([ESA_qkv(self.dim[i],heads=heads, dim_head=dim_head, dropout=dropout)
                                     for i in range(self.num)])

        self.ESAlayer = nn.ModuleList([ESA_layer(self.dim[i], heads=heads, dim_head=dim_head, dropout=dropout)
                                       for i in range(self.num)])

        self.ff = nn.ModuleList([PreNorm(self.dim[i], FeedForward(self.dim[i], mlp_dim, dropout=dropout))
                                 for i in range(self.num)])

        self.cat_kv =Cat_kv([512,512,512,512])

    def forward(self, f):
        H_list = []
        q_list = []
        k_list = []
        v_list = []
        out_list = []
        out_list_1 = []
        # num = len(f)
        for i in range(self.num):
            out = rearrange(f[i], 'b c h w -> b (h w) c')
            q,k,v = self.ESAqkv[i](f[i])
            b, c, h, w = f[i].size()
            H_list.append(h)
            q_list.append(q)
            k_list.append(k)
            v_list.append(v)
            out_list.append(out)
        k_list = self.cat_kv(k_list)
        v_list = self.cat_kv(v_list)

        for i in range(self.num):

            out = self.ESAlayer[i](q_list[i],k_list[i],v_list[i]) + out_list[i]
            out = self.ff[i](out) + out
            out = rearrange(out, 'b (h w) c -> b c h w', h=H_list[i])
            # out = rearrange(out, 'b (h w) c -> b c h w')
            out_list_1.append(out)
        return out_list_1


class PyramidVisionTransformerV2(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4],qkv_bias=False,
                 qk_scale=None, drop_rate=0.,attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], num_stages=4, linear=False,aux=True):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        self.aux =  aux

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], linear=linear)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)


        for i in range(num_stages-1,0,-1):
            patch_embed_up = OverlapPatchEmbed_up(img_size = img_size // (2 ** (i + 2)),
                                            patch_size = 3,
                                            stride = 2,
                                            in_chans  =  embed_dims[i],
                                            embed_dim = embed_dims[i-1])

            block_up = nn.ModuleList([Block(
                dim=embed_dims[i-1], num_heads=num_heads[i-1], mlp_ratio=mlp_ratios[i-1], qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0., norm_layer=norm_layer,
                sr_ratio=sr_ratios[i-1], linear=linear)
                for j in range(depths[i-1])])
            norm_up = norm_layer(embed_dims[i-1])
            cur += depths[i-1]

            change_channal = nn.Conv2d(embed_dims[i - 1]*2, embed_dims[i - 1], kernel_size=1, stride=1)

            norm_1 = nn.LayerNorm(embed_dims[i-1])
            setattr(self, f"patch_embed_up{i}", patch_embed_up)
            setattr(self, f"block_up{i}", block_up)
            setattr(self, f"norm_up{i}", norm_up)
            setattr(self, f"norm_1{i}", norm_1)
            setattr(self, f"change_channal{i}", change_channal)


        self.middle = CCFT()
        self.middle_CCA = CCFA()

        self.Conv_X4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(embed_dims[0], embed_dims[0], 3, padding=1),
            nn.BatchNorm2d(embed_dims[0]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(embed_dims[0], num_classes, 1),

        )


        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head


    def forward_features(self, x):
        B = x.shape[0]
        size = x.size()[2:]

        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)

        encode_feature = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            encode_feature.append(x)

        encode_feature_1 = encode_feature
        encode_feature_1 = self.middle_CCA(encode_feature_1)


        x1 = encode_feature[0]
        x2 = encode_feature[1]
        x3 = encode_feature[2]
        x4 = encode_feature[3]
        x1_1,x2_2,x3_3,x4_4,a = self.middle(x1,x2,x3,x4)
        encode_feature[0] = x1_1
        encode_feature[1] = x2_2
        encode_feature[2] = x3_3
        encode_feature[3] = x4_4
        #
        encode_feature[0] = (x1_1 + encode_feature_1[0])
        encode_feature[1] = (x2_2 + encode_feature_1[1])
        encode_feature[2] = (x3_3 + encode_feature_1[2])
        encode_feature[3] = (x4_4 + encode_feature_1[3])


        x = encode_feature[3]

        auxs=[x.view(B, 1,-1, 7, 7).mean(dim=2)]
        aux_out=[]
        for i in range(self.num_stages-1,0,-1):
            patch_embed_up = getattr(self, f"patch_embed_up{i}")
            block_up = getattr(self, f"block_up{i}")
            norm_up = getattr(self, f"norm_up{i}")
            norm_1 = getattr(self,f"norm_1{i}")


            change_channal = getattr(self, f"change_channal{i}")

            x, H, W = patch_embed_up(x)
            x = torch.cat((x,encode_feature[i-1]),dim=1)
            x = change_channal(x)

            x = x.flatten(2).transpose(1, 2)
            x = norm_1(x)
            for blk in block_up:
                x = blk(x, H, W)

            x = norm_up(x)

            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

            if self.aux:
                aux = x.view(B, 1,-1, H, W).mean(dim=2)
                auxs.append(aux)

        for a in auxs:

            a = F.interpolate(a, size, mode='bilinear', align_corners=True)
            a = a.squeeze()
            aux_out.append(a)

        x = self.Conv_X4(x)

        return x,aux_out

    def forward(self, x):
        x,aux_out = self.forward_features(x)

        return x,aux_out


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


def _conv_filter(state_dict, patch_size=16):

    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict




def CFATransUnet(pretrained=False,num_classes=2,**kwargs):
    model = PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_classes=num_classes,num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 9, 3], sr_ratios=[8, 4, 2, 1],**kwargs)
    model.default_cfg = _cfg()

    if pretrained:
        pretrained_path = r"..\pvt_v2_b3.pth"
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pretrained_dict = torch.load(pretrained_path, map_location=device)
        model_dict = model.state_dict()
        full_dict = copy.deepcopy(pretrained_dict)
        for k, v in pretrained_dict.items():
            if "block" in k:
                current_k = "block_up" + k[5:]
                full_dict.update({current_k: v})
        for k in list(full_dict.keys()):
            if k in model_dict:
                if full_dict[k].shape != model_dict[k].shape:
                    del full_dict[k]

        model.load_state_dict(full_dict, strict=False)

    return model



if __name__ == '__main__':

    # input = torch.rand(1, 3, 224, 224)
    input = torch.rand(1, 1, 224, 224)
    net = CFATransUnet(pretrained=True,num_classes=9)

    model_dict = net.state_dict()
    output,atten = net(input)
    print(output.size())

    from thop import profile
    from thop import clever_format
    Macs, params = profile(net, inputs=(input, ))
    flops, params = clever_format([Macs*2, params], "%.3f")
    print('flops: ', flops)
    print('params: ',params)







