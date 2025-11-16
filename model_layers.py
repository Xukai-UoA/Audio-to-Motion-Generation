import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import math
import pdb

import torch
import torch.nn as nn
from transformers import BertModel
import logging

logging.getLogger('transformers').setLevel(logging.CRITICAL)


def num_powers_of_two(x):
    num_powers = 0
    while x > 1:
        if x % 2 == 0:
            x /= 2
            num_powers += 1
        else:
            break
    return num_powers


def next_multiple_power_of_two(x, power=5):
    curr_power = num_powers_of_two(x)
    if curr_power < power:
        x = x * (2 ** (power - curr_power))
    return x


"""
整合卷积（Conv）、归一化（Norm）、激活函数（ReLU/Dropout）为一个模块，简化模型定义
Parameter:
    - in_channels：输入通道数。
    - out_channels：输出通道数。
    - type：卷积类型，'1d' 或 '2d'。
    - leaky：是否使用带泄露的 ReLU（LeakyReLU）。
    - downsample：是否下采样（通过调整 kernel_size 和 stride）。
    - kernel_size：卷积核大小（默认根据 downsample 自动设置）。
    - stride：卷积步长（默认根据 downsample 自动设置）。
    - padding：填充值（自动计算，无需手动设置）。
    - p：Dropout 的丢弃概率。
    - groups：分组卷积的组数。
"""
class ConvNormRelu(nn.Module):
    """双卷积模块（Conv -> BatchNormal -> ReLU）"""
    def __init__(self, in_channels, out_channels, type='1d', leaky=False, downsample=False,
                 kernel_size=None, stride=None, padding=None, p=0, groups=1):
        super(ConvNormRelu, self).__init__()

        # 若未手动指定 kernel_size 和 stride
        #     - 不下采样：使用 3x3 卷积核，步长 1
        #     - 下采样：使用 4x4 卷积核，步长 2，实现特征图尺寸减半（常见于残差网络）
        if kernel_size is None and stride is None:
            if not downsample:
                kernel_size = 3
                stride = 1
            else:
                kernel_size = 4
                stride = 2

        if padding is None:
            # 1D 卷积：当 stride 是元组（多维步长）时，逐维度计算 padding
            if isinstance(kernel_size, int) and isinstance(stride, tuple):
                padding = tuple(int((kernel_size - st) / 2) for st in stride)
            # 2D 卷积：当 kernel_size 是元组（如 (3,3)）时，逐维度计算 padding
            elif isinstance(kernel_size, tuple) and isinstance(stride, int):
                padding = tuple(int((ks - stride) / 2) for ks in kernel_size)
            # 错误处理：若 kernel_size 和 stride 维度不匹配，抛出断言错误
            elif isinstance(kernel_size, tuple) and isinstance(stride, tuple):
                assert len(kernel_size) == len(
                    stride), 'dims in kernel_size are {} and stride are {}. They must be the same'.format(
                    len(kernel_size), len(stride))
                padding = tuple(int((ks - st) / 2) for ks, st in zip(kernel_size, kernel_size))
            else:
                padding = int((kernel_size - stride) / 2)

        """
        分组卷积通道调整: PyTorch 的分组卷积要求输入/输出通道数为 groups 的倍数。
        - 将 in_channels 和 out_channels 乘以 groups，使得用户无需手动调整通道数。例如：
            * 输入通道 C，组数 G → 每组输入通道为 C/G。
            * 输出通道 D，组数 G → 每组输出通道为 D/G。
        """
        in_channels = in_channels * groups
        out_channels = out_channels * groups
        # Initialize Conv layer and Normalized Layer
        if type == '1d':
            self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=kernel_size, stride=stride, padding=padding,
                                  groups=groups)
            self.norm = nn.BatchNorm1d(out_channels)
            self.dropout = nn.Dropout(p=p)
        elif type == '2d':
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=kernel_size, stride=stride, padding=padding,
                                  groups=groups)
            self.norm = nn.BatchNorm2d(out_channels)
            self.dropout = nn.Dropout2d(p=p)

        # 默认使用 ReLU，若 leaky=True，则使用带斜率（0.2）的 LeakyReLU
        if leaky:
            self.relu = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.relu = nn.ReLU()

    def forward(self, x, **kwargs):
        """
        Execution order:


        """
        return self.relu(self.norm(self.dropout(self.conv(x))))


class SelfAttention(nn.Module):
    """
    Self-Attention mechanism to focus on important features.
    """
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # B: batch size, C: channels, T: time steps
        B, C, T = x.size()
        query = self.query_conv(x).view(B, -1, T).permute(0, 2, 1)  # (B, T, C')
        key = self.key_conv(x).view(B, -1, T)  # (B, C', T)
        value = self.value_conv(x).view(B, -1, T)  # (B, C, T)

        attention = torch.bmm(query, key)  # (B, T, T)
        attention = torch.nn.functional.softmax(attention, dim=-1)

        out = torch.bmm(attention, value.permute(0, 2, 1))  # (B, T, C)
        out = out.permute(0, 2, 1).view(B, C, T)  # (B, C, T)

        return self.gamma * out + x  # Residual connection


class ChannelAttention(nn.Module):
    """
    通道注意力机制 - 增强重要通道的特征响应
    输入输出形状: (batch_size, channels, time_steps)
    """

    def __init__(self, channel, reduction=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 平均池化分支
        avg_out = self.fc(self.avg_pool(x).squeeze(-1))
        # 最大池化分支
        max_out = self.fc(self.max_pool(x).squeeze(-1))
        # 合并注意力权重
        channel_att = (avg_out + max_out).unsqueeze(-1)
        return x * channel_att


class ResBlock(nn.Module):
    """Residual Block for adding depth without vanishing gradients."""
    def __init__(self, channels, type='1d', p=0.1):
        super(ResBlock, self).__init__()
        self.conv1 = ConvNormRelu(channels, channels, type=type, leaky=True, p=p)
        self.conv2 = ConvNormRelu(channels, channels, type=type, leaky=True, p=p)
        self.attention = SelfAttention(channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention(x)
        return x + residual


class ConvTranspose1D(nn.Module):
    """
    1. 使用 ConvTranspose2d（反卷积）实现特征图尺寸翻倍。
    2. 可选双线性插值（nn.Upsample）替代反卷积，但反卷积能学习参数，更适合复杂特征恢复
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# 通过多层卷积逐步提取音频特征，并压缩时间,频率轴，降低数据维度，同时保留关键信息
class AudioEncoder(nn.Module):
    '''
    input_shape:  (batch_size, input_channel, time_step, frequency)  # ([4, 1, 64, 128])
    output_shape: (batch_size, 256, time_step)  # ([4, 256, 64])

    1. 分层特征抽象
        - 低层卷积（Conv1-2）：提取局部时频模式（如基频、短时能量）。
        - 中层卷积（Conv3-4）：捕捉更广范围的上下文信息（如音素组合）。
        - 高层卷积（Conv5-8）：建模全局语义（如完整词汇或语音段）。
        - Conv7：保持时间步长8不变，仅微调特征。
        - Conv8：使用非方形卷积核 (3,8)：自适应频率维度压缩：针对频率轴的特定压缩比例（16→15）。增加方向敏感性：长卷积核（8）可能捕捉频率轴上的长程依赖。

    2. 时间维度压缩
        - 下采样策略：通过步幅2的卷积逐步减少时间步长（64 → 32 → 16 → 8），目的是：
        - 降低计算复杂度：减少后续层参数量和计算量。
        - 增强时间不变性：模糊时间位置差异，提升模型对时间偏移的鲁棒性。
        - 聚焦关键信息：过滤冗余细节（如噪声），保留语义核心部分。原始时间步长 64 可能过长，导致模型难以捕捉长期依赖.压缩到 8 个时间步后，模型可以更高效地学习长时语义（如整个单词或短语的结构）

    3. 频率维度压缩
        - 初始频率维度为128（对应梅尔频谱的频率带），最终压缩到15，原因包括：
        - 减少过拟合风险：高频细节可能对噪声敏感，压缩后保留判别性特征。
        - 适配下游任务：例如语音识别中，高频信息可能不如中低频重要。

    4. 通道数递增
        - 通道数从1逐步增至256，实现特征多样性爆炸：
        - 低通道数（1→64）：学习基础特征。
        - 高通道数（256）：组合多尺度特征，增强表征能力。

    '''

    def __init__(self, output_feats=64, input_channels=1, kernel_size=None, stride=None, p=0, groups=1):
        super(AudioEncoder, self).__init__()
        self.conv = nn.ModuleList([])
        self.conv.append(ConvNormRelu(input_channels, 64, type='2d', leaky=True, downsample=True,
                                      kernel_size=kernel_size, stride=stride, p=p, groups=groups))
        self.conv.append(ConvNormRelu(64, 128, type='2d', leaky=True, downsample=True,
                                      kernel_size=kernel_size, stride=stride, p=p, groups=groups))
        self.conv.append(ConvNormRelu(128, 256, type='2d', leaky=True, downsample=True,
                                      kernel_size=kernel_size, stride=stride, p=p, groups=groups))
        self.conv.append(ConvNormRelu(256, 512, type='2d', leaky=True, downsample=False,
                                      kernel_size=kernel_size, stride=stride, p=p, groups=groups))
        self.conv.append(ConvNormRelu(512, 256, type='2d', leaky=True, downsample=False,
                                      kernel_size=(3, 8), stride=1, p=p, groups=groups))

        ## deprecated, but kept only for older models
        # self.upconv = nn.Upsample(size=(output_feats, 1), mode='bilinear')
        ## TODO maybe the size should be (output_feats,1) instead, as we want to upsample along the time dimension

    def forward(self, x, time_steps=None):
        if time_steps is None:
            time_steps = x.shape[-2]  ## assume it is same as the input time steps

        x = x.unsqueeze(dim=1)  # [B, 1, T, F]

        x = nn.Sequential(*self.conv)(x)  # torch.Size([129, 256, 8, 15])
        # x = self.upconv(x)
        # 恢复时间维度：将压缩后的特征图时间序列长度(8)调整为用户指定的 time_steps
        # 双线性插值是一种基于周围像素值的加权平均方法，适用于连续图像数据，能够生成较为平滑的结果
        x = torch.nn.functional.interpolate(x, size=(time_steps, 1), mode='bilinear')  # torch.Size([129, 256, 64, 1])
        # 空间压缩：将频率维度15压缩为1
        x = x.squeeze(dim=-1)  # torch.Size([129, 256, 64])
        return x


class UNet1D(nn.Module):
    '''
    UNet model for 1D inputs

    Arguments
        input_shape: (batch_size, input_channels, time_step)  # ([128, 256, 64])
        input_channels (int): input channel size
        output_channels (int): output channel size (or the number of output features to be predicted)
        max_depth (int, optional): depth of the UNet (default: ``4``).
        kernel_size (int, optional): size of the kernel for each convolution (default: ``None``)
        stride (int, optional): stride of the convolution layers (default: ``None``)

    Inputs
        x (torch.Tensor): speech signal in form of a 3D Tensor
        x_shape: (batch_size, input_channels, time_step)  # ([128, 256, 64])

    Outputs
        x (torch.Tensor): input transformed to a lower frequency
            latent vector
    '''
    def __init__(self, input_channels, output_channels, max_depth=5, kernel_size=None, stride=None, p=0, groups=1):
        super(UNet1D, self).__init__()
        self.downsample_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        self.bottleneck = nn.ModuleList()
        self.max_depth = max_depth

        # 下采样路径
        self.downsample_layers.append(ConvNormRelu(in_channels=input_channels, out_channels=input_channels*2, type='1d', leaky=True, downsample=False,
                                      kernel_size=kernel_size, stride=stride, p=p, groups=groups))
        self.downsample_layers.append(ConvNormRelu(in_channels=input_channels*2, out_channels=input_channels*2, type='1d', leaky=True, downsample=True,
                                        kernel_size=kernel_size, stride=stride, p=p, groups=groups))
        self.downsample_layers.append(ConvNormRelu(in_channels=input_channels*2, out_channels=input_channels*4, type='1d', leaky=True, downsample=False,
                                        kernel_size=kernel_size, stride=stride, p=p, groups=groups))
        self.downsample_layers.append(ConvNormRelu(in_channels=input_channels*4, out_channels=input_channels*4, type='1d', leaky=True, downsample=True,
                                        kernel_size=kernel_size, stride=stride, p=p, groups=groups))

        # 瓶颈层
        self.bottleneck = ConvNormRelu(in_channels=input_channels*4, out_channels=input_channels*8,
            type='1d', leaky=True, downsample=False, p=p, groups=groups)

        # 上采样路径
        # self.upsample_layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        self.upsample_layers.append(ConvNormRelu(in_channels=input_channels*8, out_channels=input_channels*8, type='1d', leaky=True, downsample=False,
                                        kernel_size=kernel_size, stride=stride, p=p, groups=groups))
        self.upsample_layers.append(ConvTranspose1D(in_channels=input_channels*8, out_channels=input_channels*4, stride=2, output_padding=1))
        self.upsample_layers.append(ConvNormRelu(in_channels=input_channels * 8, out_channels=input_channels * 4, type='1d', leaky=True, downsample=False,
                                        kernel_size=kernel_size, stride=stride, p=p, groups=groups))
        self.upsample_layers.append(ConvTranspose1D(in_channels=input_channels * 4, out_channels=input_channels * 2, stride=2, output_padding=1))
        self.upsample_layers.append(ConvNormRelu(in_channels=input_channels * 4, out_channels=input_channels * 2, type='1d', leaky=True, downsample=False,
                                        kernel_size=kernel_size, stride=stride, p=p, groups=groups))

        # 最终输出层
        self.final_conv = nn.Conv1d(input_channels*2, output_channels, kernel_size=1)

        # self-Attentions mech
        self.down_attentions = nn.ModuleList([
            SelfAttention(input_channels * 2),  # 第一下采样块后
            SelfAttention(input_channels * 4),  # 第二下采样块后
        ])

        self.bottleneck_attention = SelfAttention(input_channels * 8)

        # self-Attentions at skip connection
        self.up_attentions = nn.ModuleList([
            SelfAttention(input_channels * 8),  # 第一跳跃连接后（cat后通道为input_channels*4 + input_channels*4 = input_channels*8）
            SelfAttention(input_channels * 4)  # 第二跳跃连接后（cat后通道为input_channels*2 + input_channels*2 = input_channels*4）
        ])

    def forward(self, x):
        # input_shape: (batch_size=256, input_channels=256, time_step=64)
        skip_connections = []
        down_attn_idx = 0
        up_attn_idx = 0

        # 下采样
        for i in range(0, len(self.downsample_layers), 2):
            x = self.downsample_layers[i](x)
            skip_connections.append(x)  # 保存跳跃连接

            x = self.downsample_layers[i + 1](x)
            x = self.down_attentions[down_attn_idx](x)  # 应用自注意力
            down_attn_idx += 1

        # 瓶颈层
        x = self.bottleneck(x)  # torch.Size([129, 2048, 8]) => torch.Size([129, 4096, 8])
        x = self.bottleneck_attention(x)
        """
        # 上采样
        for i in range(0, len(self.upsample_layers), 2):
            x = self.upsample_layers[i](x)  # 上采样或卷积

            if i < len(self.upsample_layers) - 1:  # 非最后一层
                x = self.upsample_layers[i + 1](x)  # 下一层操作

            if i % 2 == 0 and i > 0:  # 在上采样块后应用注意力
                skip_connection = skip_connections.pop()
                x = torch.cat([x, skip_connection], dim=1)
                x = self.up_attentions[up_attn_idx](x)  # 应用自注意力
                up_attn_idx += 1
        """

        # 上采样
        for i in range(0, len(self.upsample_layers), 2):
            if i % 2 == 0 and i > 0:  # cat and attention before post-cat conv
                skip_connection = skip_connections.pop()
                x = torch.cat([x, skip_connection], dim=1)
                x = self.up_attentions[up_attn_idx](x)  # Apply attention on cat result
                up_attn_idx += 1

            x = self.upsample_layers[i](x)  # post-cat conv for i>0

            if i < len(self.upsample_layers) - 1:
                x = self.upsample_layers[i + 1](x)  # Next transpose (if applicable)

        return self.final_conv(x)


class UNet1D_first_version(nn.Module):

    def __init__(self, input_channels, output_channels, max_depth=5, kernel_size=None, stride=None, p=0, groups=1):
        super(UNet1D_first_version, self).__init__()
        self.pre_downsampling_conv = nn.ModuleList([])
        self.conv1 = nn.ModuleList([])
        self.conv2 = nn.ModuleList([])
        self.upconv = nn.Upsample(scale_factor=2, mode='nearest')
        self.max_depth = max_depth
        self.groups = groups

        ## pre-downsampling
        self.pre_downsampling_conv.append(ConvNormRelu(input_channels, output_channels,
                                                       type='1d', leaky=True, downsample=False,
                                                       kernel_size=kernel_size, stride=stride, p=p, groups=groups))
        self.pre_downsampling_conv.append(ConvNormRelu(input_channels, output_channels,
                                                       type='1d', leaky=True, downsample=False,
                                                       kernel_size=kernel_size, stride=stride, p=p, groups=groups))
        for i in range(self.max_depth):
            self.conv1.append(ConvNormRelu(input_channels, output_channels,
                                           type='1d', leaky=True, downsample=True,
                                           kernel_size=kernel_size, stride=stride, p=p, groups=groups))

        for i in range(self.max_depth):
            self.conv2.append(ConvNormRelu(input_channels, output_channels,
                                           type='1d', leaky=True, downsample=False,
                                           kernel_size=kernel_size, stride=stride, p=p, groups=groups))

    def forward(self, x, return_bottleneck=False):
        input_size = x.shape[-1]
        assert input_size / (2 ** (self.max_depth - 1)) >= 1, 'Input size is {}. It must be >= {}'.format(input_size, 2**(self.max_depth - 1))
        # assert np.log2(input_size) == int(np.log2(input_size)), 'Input size is {}. It must be a power of 2.'.format(input_size)
        assert num_powers_of_two(
            input_size) >= self.max_depth, 'Input size is {}. It must be a multiple of 2^(max_depth) = 2^{} = {}'.format(
            input_size, self.max_depth, 2 ** self.max_depth)

        x = nn.Sequential(*self.pre_downsampling_conv)(x)

        residuals = []
        residuals.append(x)
        for i, conv1 in enumerate(self.conv1):
            x = conv1(x)
            if i < self.max_depth - 1:
                residuals.append(x)

        bn = x
        for i, conv2 in enumerate(self.conv2):
            x = self.upconv(x) + residuals[self.max_depth - i - 1]
            x = conv2(x)

        if return_bottleneck:
            return x, bn
        else:
            return x


class PoseEncoder(nn.Module):
    '''
    input_shape:  (N, time, pose_features: 104) #changed to 96?
    output_shape: (N, 256, time)
    '''

    def __init__(self, output_feats=64, input_channels=96, kernel_size=None, stride=None, p=0, groups=1):
        super(PoseEncoder, self).__init__()
        self.conv = nn.ModuleList([])
        self.conv.append(ConvNormRelu(input_channels, 64, type='1d', leaky=True, downsample=False,
                                      kernel_size=kernel_size, stride=stride, p=p, groups=groups))
        self.conv.append(ConvNormRelu(64, 64, type='1d', leaky=True, downsample=False,
                                      kernel_size=kernel_size, stride=stride, p=p, groups=groups))

        self.conv.append(ConvNormRelu(64, 128, type='1d', leaky=True, downsample=False,
                                      kernel_size=kernel_size, stride=stride, p=p, groups=groups))
        self.conv.append(ConvNormRelu(128, 128, type='1d', leaky=True, downsample=False,
                                      kernel_size=kernel_size, stride=stride, p=p, groups=groups))

        self.conv.append(ConvNormRelu(128, 256, type='1d', leaky=True, downsample=False,
                                      kernel_size=kernel_size, stride=stride, p=p, groups=groups))
        self.conv.append(ConvNormRelu(256, 256, type='1d', leaky=True, downsample=False,
                                      kernel_size=kernel_size, stride=stride, p=p, groups=groups))

        ## deprecated, but kept only for older models
        # self.upconv = nn.Upsample(size=(output_feats, 1), mode='bilinear')
        ## TODO maybe the size should be (output_feats,1) instead, as we want to upsample along the time dimension

    def forward(self, x, time_steps=None):
        x = torch.transpose(x, 1, 2)
        if time_steps is None:
            time_steps = x.shape[-2]  ## assume it is same as the input time steps

        x = nn.Sequential(*self.conv)(x)
        # x = self.upconv(x)
        # x = torch.nn.functional.interpolate(x, size=(time_steps, 1), mode='bilinear')
        x = x.squeeze(dim=-1)
        return x

        ## deprecated, but kept only for older models
        # self.upconv = nn.Upsample(size=(output_feats, 1), mode='bilinear')
        ## TODO maybe the size should be (output_feats,1) instead, as we want to upsample along the time dimension


class PoseStyleEncoder(nn.Module):
    '''
    input_shape:  (N, time, pose_features: 104) #changed to 96?
    output_shape: (N, 256, t)
    '''

    def __init__(self, output_feats=64, input_channels=96, kernel_size=None, stride=None, p=0, groups=1,
                 num_speakers=4):
        super().__init__()
        self.conv = nn.ModuleList([])
        self.conv.append(ConvNormRelu(input_channels, 64, type='1d', leaky=True, downsample=False,
                                      kernel_size=kernel_size, stride=stride, p=p, groups=groups))
        self.conv.append(ConvNormRelu(64, 64, type='1d', leaky=True, downsample=True,
                                      kernel_size=kernel_size, stride=stride, p=p, groups=groups))

        self.conv.append(ConvNormRelu(64, 128, type='1d', leaky=True, downsample=True,
                                      kernel_size=kernel_size, stride=stride, p=p, groups=groups))
        self.conv.append(ConvNormRelu(128, 128, type='1d', leaky=True, downsample=True,
                                      kernel_size=kernel_size, stride=stride, p=p, groups=groups))

        self.conv.append(ConvNormRelu(128, 256, type='1d', leaky=True, downsample=True,
                                      kernel_size=kernel_size, stride=stride, p=p, groups=groups))
        self.conv.append(ConvNormRelu(256, 256, type='1d', leaky=True, downsample=True,
                                      kernel_size=kernel_size, stride=stride, p=p, groups=groups))

        self.conv.append(ConvNormRelu(256, num_speakers, type='1d', leaky=True, downsample=True,
                                      kernel_size=kernel_size, stride=stride, p=p, groups=groups))

        ## deprecated, but kept only for older models
        # self.upconv = nn.Upsample(size=(output_feats, 1), mode='bilinear')
        ## TODO maybe the size should be (output_feats,1) instead, as we want to upsample along the time dimension

    def forward(self, x, time_steps=None):
        x = torch.transpose(x, 1, 2)
        if time_steps is None:
            time_steps = x.shape[-2]  ## assume it is same as the input time steps

        x = nn.Sequential(*self.conv)(x)
        # x = self.upconv(x)
        # x = torch.nn.functional.interpolate(x, size=(time_steps, 1), mode='bilinear')
        x = x.mean(-1)
        x = x.squeeze(dim=-1)
        return x


class PoseDecoder(nn.Module):
    '''
    input_shape:  (N, channels, time)
    output_shape: (N, 256, time)
    '''

    def __init__(self, input_channels=256, style_dim=10, num_clusters=8, out_feats=96, kernel_size=None, stride=None,
                 p=0):
        super().__init__()
        self.num_clusters = num_clusters
        self.style_dim = style_dim
        self.pose_decoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(input_channels + style_dim,
                                                                       input_channels,
                                                                       type='1d', leaky=True, downsample=False,
                                                                       p=p, groups=num_clusters)
                                                          for i in range(4)]))
        self.pose_logits = nn.Conv1d(input_channels * self.num_clusters, out_feats * self.num_clusters, kernel_size=1,
                                     stride=1, groups=self.num_clusters)

    def forward(self, x, **kwargs):
        style = x.view(x.shape[0], -1, self.num_clusters, x.shape[-1])[:, -self.style_dim:]
        for i, model in enumerate(self.pose_decoder):
            # x = torch.split(x, int(x.shape[1]/self.num_clusters), dim=1)
            # x = torch.cat([torch.cat([x_, kwargs['style']], dim=1) for x_ in x], dim=1)
            x = model(x)
            if i < len(self.pose_decoder) - 1:  ## last module
                x = x.view(x.shape[0], -1, self.num_clusters, x.shape[-1])
                x = torch.cat([x, style], dim=1).view(x.shape[0], -1, x.shape[-1])
        return self.pose_logits(x)


class StyleDecoder(nn.Module):
    '''
    input_shape:  (N, channels, time)
    output_shape: (N, 256, time)
    '''

    def __init__(self, input_channels=256, num_clusters=10, out_feats=96, kernel_size=None, stride=None, p=0):
        super().__init__()
        self.num_clusters = num_clusters
        self.pose_decoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(input_channels,
                                                                       input_channels,
                                                                       type='1d', leaky=True, downsample=False,
                                                                       p=p, groups=num_clusters)
                                                          for i in range(2)]))
        self.pose_logits = nn.Conv1d(input_channels * self.num_clusters, out_feats * self.num_clusters, kernel_size=1,
                                     stride=1, groups=self.num_clusters)

    def forward(self, x, **kwargs):
        x = self.pose_decoder(x)
        return self.pose_logits(x)


# TODO Unify Encoders via input_channel size?
class TextEncoder1D(nn.Module):
    '''
    input_shape:  (N, time, text_features: 300)
    output_shape: (N, 256, time)
    '''

    def __init__(self, output_feats=64, input_channels=300, kernel_size=None, stride=None, p=0, groups=1):
        super().__init__()
        self.conv = nn.ModuleList([])

        self.conv.append(ConvNormRelu(input_channels, 64, type='1d', leaky=True, downsample=False,
                                      kernel_size=kernel_size, stride=stride, p=p, groups=groups))
        self.conv.append(ConvNormRelu(64, 64, type='1d', leaky=True, downsample=False,
                                      kernel_size=kernel_size, stride=stride, p=p, groups=groups))

        self.conv.append(ConvNormRelu(64, 128, type='1d', leaky=True, downsample=False,
                                      kernel_size=kernel_size, stride=stride, p=p, groups=groups))
        self.conv.append(ConvNormRelu(128, 128, type='1d', leaky=True, downsample=False,
                                      kernel_size=kernel_size, stride=stride, p=p, groups=groups))

        self.conv.append(ConvNormRelu(128, 256, type='1d', leaky=True, downsample=False,
                                      kernel_size=kernel_size, stride=stride, p=p, groups=groups))
        self.conv.append(ConvNormRelu(256, 256, type='1d', leaky=True, downsample=False,
                                      kernel_size=kernel_size, stride=stride, p=p, groups=groups))

    def forward(self, x, time_steps=None, **kwargs):
        x = torch.transpose(x, 1, 2)
        if time_steps is None:
            time_steps = x.shape[-2]  ## assume it is same as the input time steps

        x = nn.Sequential(*self.conv)(x)
        # x = self.upconv(x)
        # x = torch.nn.functional.interpolate(x, size=(time_steps, 1), mode='bilinear')
        x = x.squeeze(dim=-1)
        return x


class Transpose(nn.Module):
    def __init__(self, idx):
        super().__init__()
        self.param = torch.nn.Parameter(torch.ones(1))
        self.idx = idx

    def forward(self, x, *args, **kwargs):
        return x.transpose(*self.idx)


class AudioEncoder1D(nn.Module):
    '''
    input_shape:  (N, time, audio_features: 128)
    output_shape: (N, 256, output_feats)
    '''

    def __init__(self, output_feats=64, input_channels=128, kernel_size=None, stride=None, p=0, groups=1):
        super().__init__()
        self.conv = nn.ModuleList([])
        self.conv.append(ConvNormRelu(input_channels, 64, type='1d', leaky=True, downsample=False,
                                      kernel_size=kernel_size, stride=stride, p=p, groups=groups))
        self.conv.append(ConvNormRelu(64, 64, type='1d', leaky=True, downsample=False,
                                      kernel_size=kernel_size, stride=stride, p=p, groups=groups))

        self.conv.append(ConvNormRelu(64, 128, type='1d', leaky=True, downsample=False,
                                      kernel_size=kernel_size, stride=stride, p=p, groups=groups))
        self.conv.append(ConvNormRelu(128, 128, type='1d', leaky=True, downsample=False,
                                      kernel_size=kernel_size, stride=stride, p=p, groups=groups))

        self.conv.append(ConvNormRelu(128, 256, type='1d', leaky=True, downsample=False,
                                      kernel_size=kernel_size, stride=stride, p=p, groups=groups))
        self.conv.append(ConvNormRelu(256, 256, type='1d', leaky=True, downsample=False,
                                      kernel_size=kernel_size, stride=stride, p=p, groups=groups))

    def forward(self, x, time_steps=None):
        # x = torch.transpose(x, 1, 2)
        if time_steps is None:
            time_steps = x.shape[-2]  ## assume it is same as the input time steps

        x = nn.Sequential(*self.conv)(x)
        # x = self.upconv(x)
        # x = torch.nn.functional.interpolate(x, size=(time_steps, 1), mode='bilinear')
        x = x.squeeze(dim=-1)
        return x

        ## deprecated, but kept only for older models
        # self.upconv = nn.Upsample(size=(output_feats, 1), mode='bilinear')
        ## TODO maybe the size should be (output_feats,1) instead, as we want to upsample along the time dimension


class LatentEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=2, p=0):
        super().__init__()
        enc1 = nn.ModuleList([ConvNormRelu(in_channels, hidden_channels,
                                           type='1d', leaky=True, downsample=False,
                                           p=p, groups=1)
                              for i in range(1)])
        enc2 = nn.ModuleList([ConvNormRelu(hidden_channels, hidden_channels,
                                           type='1d', leaky=True, downsample=False,
                                           p=p, groups=1)
                              for i in range(2)])
        enc3 = nn.ModuleList([ConvNormRelu(hidden_channels, out_channels,
                                           type='1d', leaky=True, downsample=False,
                                           p=p, groups=1)
                              for i in range(1)])
        self.enc = nn.Sequential(*enc1, *enc2, *enc3)

    def forward(self, x):
        x = self.enc(x)
        return x


class ClusterClassify(nn.Module):
    '''
    input_shape: (B, C, T)
    output_shape: (B, num_clusters, T)
    '''

    def __init__(self, num_clusters=8, kernel_size=None, stride=None, p=0, groups=1, input_channels=256):
        super().__init__()
        self.conv = nn.ModuleList()
        self.conv.append(ConvNormRelu(input_channels, 256, type='1d', leaky=True, downsample=False,
                                      kernel_size=kernel_size, stride=stride, p=p, groups=groups))
        self.conv += nn.ModuleList([ConvNormRelu(256, 256, type='1d', leaky=True, downsample=False,
                                                 kernel_size=kernel_size, stride=stride, p=p, groups=groups) for i in
                                    range(5)])

        self.logits = nn.Conv1d(256 * groups, num_clusters * groups, kernel_size=1, stride=1, groups=groups)

    def forward(self, x, time_steps=None):
        if time_steps is None:
            time_steps = x.shape[-2]  ## assume it is same as the input time steps

        x = nn.Sequential(*self.conv)(x)
        x = self.logits(x)
        return x


class Confidence(nn.Module):
    '''
    0 < confidence <= 1
    '''

    def __init__(self, beta=0.1, epsilon=1e-8):
        super().__init__()
        self.beta = beta
        self.epsilon = epsilon

    def forward(self, y, y_cap, confidence):
        if isinstance(confidence, int):
            confidence = torch.ones_like(y)
        sigma = self.get_sigma(confidence)
        P_YCAP_Y = self.p_ycap_y(y, y_cap, sigma)
        sigma_ycap = self.get_sigma(P_YCAP_Y)
        return self.get_entropy(sigma_ycap)

    def p_ycap_y(self, y, y_cap, sigma):
        diff = -(y - y_cap) ** 2
        diff_normalized = diff / (2 * sigma ** 2)
        prob = torch.exp(diff_normalized)
        prob_normalized = prob * (1 / (2 * math.pi * sigma))
        return prob_normalized

    def get_sigma(self, confidence):
        mask = (confidence < self.epsilon).double()
        confidence = (1 - mask) * confidence + torch.ones_like(confidence) * self.epsilon * mask
        sigma = 1 / (2 * math.pi * confidence)
        return sigma

    ## entropy of a guassian
    def get_entropy(self, sigma):
        return 0.5 * (torch.log(2 * math.pi * math.e * (sigma ** 2))) * self.beta


class Repeat(nn.Module):
    def __init__(self, repeat, dim=-1):
        super().__init__()
        self.dim = dim
        self.repeat = repeat
        # self.temp = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x.repeat_interleave(self.repeat, self.dim)


class BatchGroup(nn.Module):
    '''
    Group conv networks to run in parallel
    models: list of instantiated models

    Inputs:
      x: list of list of inputs; x[group][batch], len(x) == groups, and len(x[0]) == batches
      labels: uses these labels to give a soft attention on the outputs. labels[batch], len(labels) == batches
              if labels is None, return a list of outputs
      transpose: if true, model needs a transpose of the input
    '''

    def __init__(self, models, groups=1):
        super().__init__()
        if not isinstance(models, list):
            models = [models]
        self.models = nn.ModuleList(models)
        self.groups = groups

    def index_select_outputs(self, x, labels):
        '''
        x: (B, num_clusters*out_feats, T)
        labels: (B, T, num_clusters)
        '''
        x = x.transpose(2, 1)
        x = x.view(x.shape[0], x.shape[1], self.groups, -1)
        labels = labels.view(x.shape[0], x.shape[1], x.shape[2])  ## shape consistency while sampling
        x = (x * labels.unsqueeze(-1)).sum(dim=-2)
        return x

    def forward(self, x, labels=None, transpose=True, **kwargs):
        if not isinstance(x, list):
            raise 'x must be a list'
        if not isinstance(x[0], list):
            raise 'x must be a list of lists'
        if labels is not None:
            assert isinstance(labels, list), 'labels must be a list'

        groups = len(x)
        assert self.groups == groups, 'input groups should be the same as defined groups'
        batches = len(x[0])

        x = [torch.cat(x_, dim=0) for x_ in x]  # batch
        x = torch.cat(x, dim=1)  # group

        if transpose:
            x = x.transpose(-1, -2)
        for model in self.models:
            if kwargs:
                x = model(x, **kwargs)
            else:
                x = model(x)

        is_tuple = isinstance(x, tuple)
        if labels is not None:
            assert not is_tuple, 'labels is not None does not work with is_tuple=True'
            labels = torch.cat(labels, dim=0)  # batch
            x = [self.index_select_outputs(x, labels).transpose(-1, -2)]
        else:  # separate the groups
            if is_tuple:
                channels = [int(x[i].shape[1] / groups) for i in range(len(x))]
                x = [torch.split(x_, channels[i], dim=1) for i, x_ in enumerate(x)]
                # x = list(zip(*[torch.split(x_, channels[i], dim=1) for i, x_ in enumerate(x)]))
                # x = [tuple([x_[:, start*channels[i]:(start+1)*channels[i]] for i, x_ in enumerate(x)]) for start in range(groups)]
            else:
                channels = int(x.shape[1] / groups)
                x = list(torch.split(x, channels, dim=1))
                # x = [x[:, start*channels:(start+1)*channels] for start in range(groups)]

        if is_tuple:
            channels = int(x[0][0].shape[0] / batches)
            x = tuple([[torch.split(x__, channels, dim=0) for x__ in x_] for x_ in x])
            # x = [[tuple([x__[start*channels:(start+1)*channels] for x__ in x_]) for start in range(batches)] for x_ in x]
        else:
            channels = int(x[0].shape[0] / batches)
            x = [list(torch.split(x_, channels, dim=0)) for x_ in x]
            # x = [[x_[start*channels:(start+1)*channels] for start in range(batches)] for x_ in x]
        return x


class Group(nn.Module):
    '''
    Group conv networks to run in parallel
    models: list of instantiated models
    groups: groups of inputs
    dim: if dim=0, use batch a set of inputs along batch dimension (group=1 always)
         elif dim=1, combine the channel dimension (group=num_inputs)

    Inputs:
      x: list of inputs
      labels: uses these labels to give a soft attention on the outputs. Use only with dim=1.
              if labels is None, return a list of outputs
      transpose: if true, model needs a transpose of the input
    '''

    def __init__(self, models, groups=1, dim=1):
        super().__init__()
        if not isinstance(models, list):
            models = [models]
        self.models = nn.ModuleList(models)
        self.groups = groups
        self.dim = dim

    def index_select_outputs(self, x, labels):
        '''
        x: (B, num_clusters*out_feats, T)
        labels: (B, T, num_clusters)
        '''
        x = x.transpose(2, 1)
        x = x.view(x.shape[0], x.shape[1], self.groups, -1)
        labels = labels.view(x.shape[0], x.shape[1], x.shape[2])  ## shape consistency while sampling
        x = (x * labels.unsqueeze(-1)).sum(dim=-2)
        return x

    def forward(self, x, labels=None, transpose=True, **kwargs):
        if self.dim == 0:
            self.groups = len(x)
        if isinstance(x, list):
            x = torch.cat(x, dim=self.dim)  ## concatenate along channels
        if transpose:
            x = x.transpose(-1, -2)
        for model in self.models:
            if kwargs:
                x = model(x, **kwargs)
            else:
                x = model(x)
        if labels is not None:
            x = self.index_select_outputs(x, labels).transpose(-1, -2)  ## only for dim=1
            return x
        else:
            channels = int(x.shape[self.dim] / self.groups)
            dim = self.dim % len(x.shape)
            if dim == 2:
                x = [x[:, :, start * channels:(start + 1) * channels] for start in range(self.groups)]
            elif dim == 1:
                x = [x[:, start * channels:(start + 1) * channels] for start in range(self.groups)]
            elif dim == 0:
                x = [x[start * channels:(start + 1) * channels] for start in range(self.groups)]
            return x


class EmbLin(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.emb = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x, mode='lin'):
        if mode == 'lin':
            return x.matmul(self.emb.weight)
        elif mode == 'emb':
            return self.emb(x)


class Style(nn.Module):
    '''
    input_shape: (B, )
    output_shape: (B, )
    '''

    def __init__(self, num_speakers=1):
        self.style_emb = nn.Embedding(num_embeddings=num_speakers, embedding_dim=256)

    def forward(self, x):
        pass


class Curriculum():
    def __init__(self, start, end, num_iters):
        self.start = start
        self.end = end
        self.num_iters = num_iters
        self.iters = 0
        self.diff = (end - start) / num_iters
        self.value = start

    def step(self, flag=True):
        if flag:
            value_temp = self.value
            if self.iters < self.num_iters:
                self.value += self.diff
                self.iters += 1
                return value_temp
            else:
                return self.end
        else:
            return self.value
