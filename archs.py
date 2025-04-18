
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
from torch import nn

import torch
import torch.nn as nn
import torch.nn.functional as F


from attition.Attition import CBAM
from model.backbone import build_backbone

import numpy as np
__all__ = ['UNet', 'NestedUNet']


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        # self.conv1 = DeformConv2D(in_channels, middle_channels, 3, padding=1)


        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        # self.conv2 = DeformConv2D(middle_channels, out_channels, 3, padding=1)

        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out



class UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False,**kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)#scale_factor:放大的倍数  插值

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output


class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        # print('input:',input.shape)
        x0_0 = self.conv0_0(input)
        # print('x0_0:',x0_0.shape)
        x1_0 = self.conv1_0(self.pool(x0_0))
        # print('x1_0:',x1_0.shape)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        # print('x0_1:',x0_1.shape)

        x2_0 = self.conv2_0(self.pool(x1_0))
        # print('x2_0:',x2_0.shape)
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        # print('x1_1:',x1_1.shape)
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        # print('x0_2:',x0_2.shape)

        x3_0 = self.conv3_0(self.pool(x2_0))
        # print('x3_0:',x3_0.shape)
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        # print('x2_1:',x2_1.shape)
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        # print('x1_2:',x1_2.shape)
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        # print('x0_3:',x0_3.shape)
        x4_0 = self.conv4_0(self.pool(x3_0))
        # print('x4_0:',x4_0.shape)
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        # print('x3_1:',x3_1.shape)
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        # print('x2_2:',x2_2.shape)
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        # print('x1_3:',x1_3.shape)
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        # print('x0_4:',x0_4.shape)

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dimension=2, batch_norm=True):
        super().__init__()
        Conv = nn.Conv2d if dimension == 2 else nn.Conv3d
        BatchNorm = nn.BatchNorm2d if dimension == 2 else nn.BatchNorm3d

        self.conv1 = Conv(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = BatchNorm(out_channels) if batch_norm else nn.Identity()
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = Conv(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = BatchNorm(out_channels) if batch_norm else nn.Identity()

        self.shortcut = Conv(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x += residual
        x = self.act2(x)
        return x


class ResUNet(nn.Module):
    def __init__(self, filter_root=64, depth=4, n_classes=1, input_size=(3, 512, 512),
                 activation='relu', batch_norm=True, final_activation='softmax'):
        super().__init__()
        self.depth = depth
        self.batch_norm = batch_norm
        self.dimension = len(input_size) - 1  # 2D or 3D based on input shape (C, H, W) or (C, D, H, W)

        # Determine convolution type
        Conv = nn.Conv2d if self.dimension == 2 else nn.Conv3d
        MaxPool = nn.MaxPool2d if self.dimension == 2 else nn.MaxPool3d
        self.UpSample = nn.Upsample if self.dimension == 2 else nn.Upsample

        # Down path
        self.down_blocks = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        in_channels = input_size[0]

        for i in range(depth):
            out_channels = filter_root * (2 ** i)
            block = ResidualBlock(in_channels, out_channels, self.dimension, batch_norm)
            self.down_blocks.append(block)
            if i < depth - 1:
                self.pool_layers.append(MaxPool(kernel_size=2, stride=2))
            in_channels = out_channels

        # Up path
        self.up_blocks = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        for i in range(depth - 2, -1, -1):
            out_channels = filter_root * (2 ** i)
            self.up_convs.append(
                Conv(out_channels * 2, out_channels, kernel_size=3, padding=1)
            )
            self.up_blocks.append(
                ResidualBlock(out_channels * 2, out_channels, self.dimension, batch_norm)
            )

        # Final convolution
        self.final_conv = Conv(filter_root, n_classes, kernel_size=1)

        # Activation functions
        self.final_activation = final_activation

    def forward(self, x):
        skip_connections = []

        # Down path
        for i in range(self.depth):
            x = self.down_blocks[i](x)
            if i < self.depth - 1:
                skip_connections.append(x)
                x = self.pool_layers[i](x)

        # Up path
        for i in range(self.depth - 1):
            x = self.UpSample(scale_factor=2, mode='nearest')(x)
            x = self.up_convs[i](x)
            skip = skip_connections.pop(-1)

            # Handle potential size mismatch
            diff = [skip.size(dim) - x.size(dim) for dim in range(2, x.dim())]
            pad_params = []
            for dim_diff in diff:
                pad_left = dim_diff // 2
                pad_right = dim_diff - pad_left
                pad_params.extend([pad_left, pad_right])
            x = F.pad(x, pad_params)

            x = torch.cat([x, skip], dim=1)
            x = self.up_blocks[i](x)

        # Final convolution
        x = self.final_conv(x)

        # Apply final activation
        # if self.final_activation == 'softmax':
        #     x = F.softmax(x, dim=1)
        # elif self.final_activation == 'sigmoid':
        #     x = torch.sigmoid(x)

        return x