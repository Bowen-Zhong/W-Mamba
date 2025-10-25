import numpy as np
import torch
import torch.nn as nn
from torchprofile import profile_macs

# from Module.attn_module import Attention, PatchEmbed, DePatch, Mlp, Block
from timm.models.layers import DropPath
import antialiased_cnns
# from Module.FourierUnit import AdaFreFusion
# from Module.ExpertsUnit import SpaFreExpFusion

from Module.ConvSSM_add_deepthconv import ConvSSM
from Module.WTConv import DepthwiseSeparableConvWithWTConv2d
from Module.gate import GatedMultimodalLayer



class W_Mamba(nn.Module):
    def __init__(self, patch_size=16, dim=256, num_heads=8, channels=[112, 160, 208, 256],
                 fusionblock_depth=[4, 4, 4, 4], qk_scale=None, attn_drop=0., proj_drop=0.):
        super(W_Mamba, self).__init__()

        self.encoder = encoder_convblock()

        self.conv_up4 = ConvBlock_up(channels[-1] * 2, 104, 208)  # ori
        self.conv_up3 = ConvBlock_up(channels[-2] * 3, 80, 160)  # ori
        self.conv_up2 = ConvBlock_up(channels[-3] * 3, 56, 112)  # ori
        self.conv_up1 = ConvBlock_up(channels[-4] * 3, 8, 16, if_up=False)  # ori

        # Fusion Block
        # self.fusionnet1 = FusionModule(patch_size=patch_size, dim=dim, num_heads=num_heads,
        #                                channels=channels[0],
        #                                fusionblock_depth=fusionblock_depth[0],
        #                                qk_scale=qk_scale, attn_drop=attn_drop,
        #                                proj_drop=proj_drop)
        # self.fusionnet2 = FusionModule(patch_size=patch_size, dim=dim, num_heads=num_heads,
        #                                channels=channels[1],
        #                                fusionblock_depth=fusionblock_depth[1],
        #                                qk_scale=qk_scale, attn_drop=attn_drop,
        #                                proj_drop=proj_drop)
        # self.fusionnet3 = FusionModule(patch_size=patch_size, dim=dim, num_heads=num_heads,
        #                                channels=channels[2],
        #                                fusionblock_depth=fusionblock_depth[2],
        #                                qk_scale=qk_scale, attn_drop=attn_drop,
        #                                proj_drop=proj_drop)
        # self.fusionnet4 = FusionModule(patch_size=patch_size, dim=dim, num_heads=num_heads,
        #                                channels=channels[3],
        #                                fusionblock_depth=fusionblock_depth[3],
        #                                qk_scale=qk_scale, attn_drop=attn_drop,
        #                                proj_drop=proj_drop)

        # Conv 1x1
        self.outlayer = nn.Conv2d(16, 1, 1)
        self.SSM1 = ConvSSM(256)
        self.SSM2 = ConvSSM(128)
        self.SSM3 = ConvSSM(64)
        self.SSM4 = ConvSSM(32)
        # self.SSM1 = ConvSSM(128)
        # self.SSM2 = ConvSSM(64)
        # self.SSM3 = ConvSSM(32)
        # self.SSM4 = ConvSSM(16)
        self.gml1 = GatedMultimodalLayer(channels_in1=112, channels_in2=112, channels_out=224)
        self.gml2 = GatedMultimodalLayer(channels_in1=160, channels_in2=160, channels_out=320)
        self.gml3 = GatedMultimodalLayer(channels_in1=208, channels_in2=208, channels_out=416)
        self.gml4 = GatedMultimodalLayer(channels_in1=256, channels_in2=256, channels_out=512)
    def forward(self, img1, img2):
        x1, x2, x3, x4 = self.encoder(img1)
        y1, y2, y3, y4 = self.encoder(img2)

        # z1 = self.fusionnet1(x1, y1)
        # z2 = self.fusionnet2(x2, y2)
        # z3 = self.fusionnet3(x3, y3)
        # z4 = self.fusionnet4(x4, y4)
        #print(x1.shape,y1.shape,x2.shape,y2.shape,x3.shape,y3.shape,x4.shape,y4.shape)

        x11 = self.SSM1(x1)
        y11 = self.SSM1(y1)

        x22 = self.SSM2(x2)
        y22 = self.SSM2(y2)

        x33 = self.SSM3(x3)
        y33 = self.SSM3(y3)

        x44 = self.SSM4(x4)
        y44 = self.SSM4(y4)

        z1 = self.gml1(x11,y11) + torch.cat((x1, y1), dim=1)
        z2 = self.gml2(x22,y22) + torch.cat((x2, y2), dim=1)
        z3 = self.gml3(x33,y33) + torch.cat((x3, y3), dim=1)
        z4 = self.gml4(x44,y44) + torch.cat((x4, y4), dim=1)
        # z1 = torch.cat((x1, y1), dim=1)
        # z2 = torch.cat((x2, y2), dim=1)
        # z3 = torch.cat((x3, y3), dim=1)
        # z4 = torch.cat((x4, y4), dim=1)

        # z1 = self.SSM1(z1)
        # z2 = self.SSM2(z2)
        # z3 = self.SSM3(z3)
        # z4 = self.SSM4(z4)

        out4 = self.conv_up4(z4)
        out3 = self.conv_up3(torch.cat((out4, z3), dim=1))
        out2 = self.conv_up2(torch.cat((out3, z2), dim=1))
        out1 = self.conv_up1(torch.cat((out2, z1), dim=1))

        img_fusion = self.outlayer(out1)

        return img_fusion


class FusionModule(nn.Module):
    def __init__(self, patch_size=16, dim=256, num_heads=8, channels=256, fusionblock_depth=3,
                 qk_scale=None, attn_drop=0., proj_drop=0., img_size=256):
        super(FusionModule, self).__init__()

        self.AFFusion1 = AdaFreFusion(channels, channels)
        self.AFFusion2 = AdaFreFusion(channels, channels)

        # Fusion Block
        self.CASFusion = CroAttSpaFusion(patch_size=patch_size, dim=dim, num_heads=num_heads, channel=channels,
                                         proj_drop=proj_drop, depth=fusionblock_depth, qk_scale=qk_scale,
                                         attn_drop=attn_drop)
        self.SFEFusion = SpaFreExpFusion(dim=channels * 2)

    def forward(self, img1, img2):
        x = img1
        y = img2
        y_f = self.AFFusion1(y)
        x_f = self.AFFusion2(x)
        feature_y = x_f + y_f
        feature_x = self.CASFusion(x, y)
        z = self.SFEFusion(torch.cat([feature_x, feature_y], dim=1))

        return z


class Conv_decoder(nn.Module):
    def __init__(self, channels=[256, 128, 64, 1]):
        super(Conv_decoder, self).__init__()
        self.decoder1 = Conv_Block(channels[0], int(channels[0] + channels[1] / 2), channels[1])
        self.decoder2 = Conv_Block(channels[1], int(channels[1] + channels[2] / 2), channels[2])
        self.decoder3 = Conv_Block(channels[2], int(channels[2] / 2), channels[3])

    def forward(self, x):
        x1 = self.decoder1(x)
        x2 = self.decoder2(x1)
        out = self.decoder3(x2)

        return out


class Conv_Block(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hid_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(hid_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

        self.norm1 = nn.BatchNorm2d(hid_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)

        self.act = nn.GELU()

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.norm1(x1)
        x2 = self.conv2(x1)
        x2 = self.norm2(x2)
        out = self.act(x2)

        return out


class ConvBlock_down(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels, kernel_size=3, stride=1, padding=1, if_down=True):
        super(ConvBlock_down, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hid_channels, kernel_size, stride, padding)
        #self.conv2 = nn.Conv2d(hid_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = DepthwiseSeparableConvWithWTConv2d(hid_channels, out_channels)

        self.bn1 = nn.BatchNorm2d(hid_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.act = nn.GELU()

        self.if_down = if_down
        self.down = nn.MaxPool2d(kernel_size=2, stride=1)
        self.down_anti = antialiased_cnns.BlurPool(in_channels, stride=2)

    def forward(self, x):
        if self.if_down:
            x = self.down(x)
            x = self.down_anti(x)
            x = self.act(x)

        x1 = self.conv1(x)
        x2 = self.bn1(x1)

        x3 = self.conv2(x2)
        x3 = self.bn2(x3)
        out = self.act(x3)

        return out


class encoder_convblock(nn.Module):
    def __init__(self):
        super(encoder_convblock, self).__init__()
        self.inlayer = nn.Conv2d(1, 64, 1)
        self.block1 = ConvBlock_down(64, 32, 112, if_down=False)
        self.block2 = ConvBlock_down(112, 56, 160)
        self.block3 = ConvBlock_down(160, 80, 208)
        self.block4 = ConvBlock_down(208, 104, 256)

    def forward(self, img):
        img = self.inlayer(img)
        x1 = self.block1(img)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        return x1, x2, x3, x4

###
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super().__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, reflection_padding,
                                padding_mode='reflect')
        self.dropout = nn.Dropout2d(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        self.is_last = is_last

    def forward(self, x):
        out = self.conv2d(x)
        if self.is_last is False:
            out = self.relu(out)
        return out
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        # out_channels_def = 16
        out_channels_def = int(in_channels / 2)  #
        Block = []
        Block += [ConvLayer(in_channels, out_channels_def, kernel_size, stride),
                  ConvLayer(out_channels_def, out_channels, kernel_size, stride)]
        self.Block = nn.Sequential(*Block)


    def forward(self, x):
        out = self.Block(x)
        return out
####
class ConvBlock_up(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels, kernel_size=3, stride=1, padding=1, if_up=True):
        super(ConvBlock_up, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, hid_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(hid_channels, out_channels, kernel_size, stride, padding)

        self.bn1 = nn.BatchNorm2d(hid_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.act = nn.GELU()

        self.if_up = if_up
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):

        x1 = self.conv1(x)
        x2 = self.bn1(x1)

        x3 = self.conv2(x2)
        x3 = self.bn2(x3)

        if self.if_up:
            out = self.act(self.up(x3))
        else:
            out = self.act(x3)

        return out


class CroAttSpaFusion(nn.Module):
    def __init__(self, patch_size, dim, num_heads, channel, proj_drop, depth, qk_scale, attn_drop):
        super(CroAttSpaFusion, self).__init__()

        self.patchembed1 = PatchEmbed(patch_size=patch_size, in_c=channel, embed_dim=dim)
        self.patchembed2 = PatchEmbed(patch_size=patch_size, in_c=channel, embed_dim=dim)

        self.QKV_Block1 = Block(dim=dim, num_heads=num_heads)
        self.QKV_Block2 = Block(dim=dim, num_heads=num_heads)

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj1 = nn.Linear(dim, dim)
        self.proj2 = nn.Linear(dim, dim)
        self.proj_drop1 = nn.Dropout(proj_drop)
        self.proj_drop2 = nn.Dropout(proj_drop)

        self.depatch = DePatch(channel=channel, embed_dim=dim, patch_size=patch_size)

    def forward(self, in_1, in_2):
        # Patch Embeding1
        in_emb1 = self.patchembed1(in_1)
        B, N, C = in_emb1.shape

        # cross self-attention Feature Extraction
        _, q1, k1, v1 = self.QKV_Block1(in_emb1)

        attn1 = (q1 @ k1.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)

        # Patch Embeding2
        in_emb2 = self.patchembed2(in_2)

        _, q2, k2, v2 = self.QKV_Block2(in_emb2)

        attn2 = (q2 @ k2.transpose(-2, -1)) * self.scale
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.attn_drop(attn2)

        # cross attention
        x_attn1 = (attn1 @ v2).transpose(1, 2).reshape(B, N, C)
        x_attn1 = self.proj1(x_attn1)
        x_attn1 = self.proj_drop1(x_attn1)

        x_attn2 = (attn2 @ v1).transpose(1, 2).reshape(B, N, C)
        x_attn2 = self.proj2(x_attn2)
        x_attn2 = self.proj_drop2(x_attn2)

        x_attn = (x_attn1 + x_attn2) / 2

        # Patch Rearrange
        ori = in_2.shape  # b,c,h,w
        out1 = self.depatch(x_attn, ori)

        out = in_1 + in_2 + out1

        return out


class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()  # ==Dropout
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x1 = self.norm1(x)
        attn_list = self.attn(x1)  # x,q,k,v
        attn = attn_list[0]
        x1 = self.drop_path(attn)
        x = x + x1

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
import time

if __name__ == '__main__':
    device = 'cuda:0'
    img1 = torch.randn(1, 1, 256, 256).to(device)
    img2 = torch.randn(1, 1, 256, 256).to(device)
    model = ASFEFusion().to(device)
    result = model(img1, img2)
    print(result.shape)

    # model = NestFuse()
    # inputs = torch.randn(8, 1, 256, 256)
     #encode = model.encoder(inputs)

    # print(encode[3].size())
    # outputs = model.decoder_train(encode)
    # print(outputs[0].size())
    flops = profile_macs(model, (img1, img2))
    print(flops/1e9)
    params = sum(p.numel() for p in model.parameters())
    print(params/1e6)
    with torch.no_grad():
        for _ in range(10):
            _ = model(img1, img2)
        torch.cuda.synchronize()  # 确保所有CUDA操作完成

    # 测试推理时间
    num_runs = 1000  # 测试次数
    timings = []

    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(img1, img2)
            torch.cuda.synchronize()  # 同步CUDA操作，确保时间准确
            end_time = time.time()
            timings.append(end_time - start_time)

    # 计算平均时间和标准差
    avg_time = sum(timings) / num_runs * 1000  # 转换为毫秒
    std_time = torch.std(torch.tensor(timings)).item() * 1000
    print(f'Average inference time: {avg_time:.2f}ms ± {std_time:.2f}ms')