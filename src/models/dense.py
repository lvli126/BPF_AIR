import torch 
import torch.nn as nn
from torch.nn.modules.activation import ReLU

# 稠密块是由多个convblock组成，因此要先定义conv block. 该块采用的是BN+ReLU+Conv3x3的结构
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
            
        )

    def forward(self, x):
        return self.conv(x)

# 定义稠密块，每个conv_block块使用相同的输出通道数。但在前向计算时，将每块的输入和输出在通道维上连结。这里的out_channels也相当于modules中的growth_rates
class DenseBlock(nn.Module):
    def __init__(self, num_convs, in_channels, out_channels):
        super(DenseBlock, self).__init__()
        net=[]
        for i in range(num_convs):
           in_c = in_channels + out_channels*i
           net.append(ConvBlock(in_c, out_channels))
        self.net = nn.ModuleList(net)

    def forward(self, x):
        for blk in self.net:
            y = blk(x)
            x = torch.cat((x,y), dim=1)
        return x

# 由于每个稠密块连结都会带来通道数的增加，使用过多则会带来过于复杂的模型。过渡层用来控制模型复杂度。它通过1×1卷积层来减小通道数，并使用步幅为2的平均池化层减半高和宽，从而进一步降低模型复杂度。
class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionBlock, self).__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(in_channels), 
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        return self.transition(x)

class DenseModule(nn.Module):
    def __init__(self, in_channels, out_channels, growth_rate = 24, num_convs = 4):
        super().__init__()
        self.in_conv = nn.Conv2d(1, in_channels, kernel_size=3, padding=1)
        self.dense_block = DenseBlock(num_convs, in_channels, growth_rate)
        self.out_conv = TransitionBlock(in_channels+growth_rate*num_convs, out_channels)

    def forward(self, x):
        return self.out_conv(self.dense_block(self.in_conv(x)))