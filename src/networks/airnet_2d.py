import torch
from torch._C import device
import torch.nn as nn
from models.tof2d import ReconLayer
from models.model import ConvSequence

class AirNet2d(nn.Module):

    def __init__(self, network_init):
        '''feature_channels表示输入到CNN的特征数据。如果只是2D或3D的PET图像，则feature_channels=1；
            如加上额外的其他模态的图像或者做4D重建，则值为1+图像的channel数
        '''
        super().__init__()
        self.s_factor = network_init.s_factor
        self.s_factor = network_init.s_factor
        self.image_para = network_init.image_para
        self.device =  network_init.device    
        self.feature_channels = network_init.feature_channels
        self.mid_channels = network_init.mid_channels
        self.num_seq_blocks =  network_init.num_seq_blocks
        self.num_iters = network_init.num_iters
        self.recon_layers = nn.ModuleList([ReconLayer(self.s_factor,self.image_para, self.device) for i in range(self.num_iters)])
        self.conv_seqs = nn.ModuleList(
            [ConvSequence(self.feature_channels+i, self.mid_channels, 1, self.num_seq_blocks, 3) for i in range(self.num_iters)]
        )
        
    def forward(self, x0, projection_data, listmode):
        # x = torch.zeros_like(x0, dtype=torch.float, device = self.device)
        x = x0.detach().clone().to(self.device)
        for i, (recon_layer, conv_seq) in enumerate(zip(self.recon_layers, self.conv_seqs)):
            x_1_2, y = recon_layer(x, projection_data, listmode)
            # print(x_1_2.shape)
            if i ==0:
                dense = x_1_2
            else:
                dense = torch.cat((dense, x_1_2), dim=1)
            x = conv_seq(dense) + x_1_2
        return x, y # y为所留投影域数据接口，用于以后做hybrid domain loss

        
