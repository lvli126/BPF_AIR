import torch
import torch.nn as nn

# CNN for penalty term
# 这一部分包含三个卷积项，前两个卷积由Conv+ReLU组成； 最后一个只有Conv（PET上也可以包含ReLU）
class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

# 组成这个CNN项
class ConvSequence(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, num_blocks, kernel_size=3):
        super().__init__()
        self.in_conv = SingleConv(in_channels, mid_channels, kernel_size)
        self.mid_blocks = nn.ModuleList(
            [SingleConv(mid_channels, mid_channels, kernel_size) for i in range(num_blocks)]
        )
        self.out_conv = nn.Conv2d(mid_channels, out_channels, kernel_size)
        # 也可以换成带relu的卷积
        # self.out_conv = SingleConv(mid_channels, out_channels, kernel_size)
    
    def forward(self, x):
        x = self.in_conv(x)
        for block in self.mid_blocks:
            x = block(x)
        x = self.out_conv(x)
        return x