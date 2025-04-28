import torch 
import torch.nn as nn
import torch.nn.functional as F
from att_module_1.temporal import temporal
from att_module_1.spatial import spatial

class module_1(nn.Module):
    def __init__(self, channels, r):
        super(module_1, self).__init__()
        self.channels = channels
        self.r = r
        self.sam = spatial(bias=True)
        self.cam = temporal(channels=self.channels, r=self.r)

        self.attention_maps = {}  # Dictionary to store the channel and spatial attention maps

    def forward(self, x):
        output_cam = self.cam(x)
        self.attention_maps['channel_weights'] = self.cam.output_sig.clone().detach().cpu().numpy()
        self.attention_maps['channel_results'] = output_cam.clone().detach().cpu().numpy()
        output_sam = self.sam(output_cam)
        self.attention_maps['spatial_weights'] = self.sam.output_sig.clone().detach().cpu().numpy()
        self.attention_maps['spatial_results'] = output_sam.clone().detach().cpu().numpy()
        return  output_sam + x 

