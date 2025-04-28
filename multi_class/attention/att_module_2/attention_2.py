import torch 
import torch.nn as nn
import torch.nn.functional as F
from att_module_2.temporal_ import temporal_2

class module_2(nn.Module):
    def __init__(self, channels, r):
        super(module_2, self).__init__()
        self.channels = channels
        self.r = r
        self.cam = temporal_2(channels=self.channels, r=self.r)

    def forward(self, x):
        output_cam = self.cam(x)
        return output_cam + x

