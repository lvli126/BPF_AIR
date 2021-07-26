import torch
import torch.nn as nn
from operators.pytorch.online2d.projector import BpMapping


class ReconLayer(nn.Module):
    # first step of air recon, called ReconLayer
    # second step of air recon, called PenalizedLayer

    def __init__(self, s_factor,image_para, cuda_device):
        super(ReconLayer, self).__init__()
        self.recon_layer = BpMapping(s_factor, image_para, cuda_device)

    def forward(self, image, projection_data, listmode):
        image_new = self.recon_layer(image, projection_data, listmode)
        return image_new, projection_data