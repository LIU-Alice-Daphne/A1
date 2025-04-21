# Purple clay pot identification
## 一、框架搭建
本项目旨在利用深度学习技术，实现紫砂壶图像的分割，帮助用户自动定位紫砂壶轮廓。整体流程涵盖：<br>
- 数据收集与处理<br>
- 分割模型构建（基于U-Net）<br>
- 模型训练与验证<br>
- 分割效果可视化与评估<br>

## 二、数据准备
1.数据收集

紫砂壶图像数据在公开平台中较为稀缺，且质量参差不齐。通过网络爬虫与人工拍摄两种方式共计收集到约200张图像，并配合以下注意事项进行筛选：

- 图像尺寸：建议原图分辨率不低于512×512
- 背景干净：尽量避免背景杂物干扰壶体识别
- 单一物体：一图一壶，避免多壶干扰分割标签

经过初步筛选与人工标注后，形成原始数据集，样本存在以下问题：

- 图像模糊：约10%图像存在焦点模糊或压缩严重现象
- 分类数量少：目前数据集中仅覆盖西施壶、石瓢壶、井栏壶、仿古壶等4类
- 结构残缺：部分图像仅包含壶体局部（如左半部分或无壶盖）

2.数据增强

为提升模型对不同姿态、光照及图像畸变的鲁棒性，采用Albumentations数据增强库进行处理，增强后图像大大缓解了样本不足的问题。增强方式包括：

- 随机旋转 ±45°
- 水平 / 垂直翻转
- 高斯模糊（适度）
- 仿射变换 / 缩放

3.数据存放格式

inputs/

├── train/

│   ├── images/

│   │   ├── 001.png    

│   │   └── ..，

│   ├── masks/

│   │   ├── 001.png    

│   │   └── ..，

├── test/

## 三、模型设计与训练

1.模型架构

本项目选用经典语义分割模型 **U-Net** 作为主干网络，结构如下：

- 编码器部分提取图像语义特征
- 解码器部分进行上采样与细节还原
- 跳跃连接融合高低层信息，增强边缘分割表现

如图所示：

![](https://i-blog.csdnimg.cn/blog\_migrate/8b0fab47a42c35a97cea1d5ebd0eb68d.png#pic\_center "8b0fab47a42c35a97cea1d5ebd0eb68d")

|**组件**|**参数/方法**|
| :-: | :-: |
|基础模型|Unet|
|损失函数|BCEDiceLoss|
|优化器|Adam|
|初始学习率|0.0001|
|Batch Size|4|
|Epoch|300|
|学习率调度策略|CosineAnnealingLR|
|权重保存策略|val\_loss最小时保存 best\_model.pth|

2.pytorch代码
```python
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
```
