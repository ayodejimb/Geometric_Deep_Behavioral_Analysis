import torch 
import torch.nn as nn
import torch.nn.functional as F

class spatial(nn.Module):
    def __init__(self, bias=False):
        super(spatial, self).__init__()
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1, dilation=1, bias=self.bias)
        self.output_sig = None

    def forward(self, x):
        max = torch.max(x,1)[0].unsqueeze(1)
        avg = torch.mean(x,1).unsqueeze(1)
        concat = torch.cat((max,avg), dim=1)
        output = self.conv(concat)
        self.output_sig = F.sigmoid(output)
        output_times_x = self.output_sig * x
        return output_times_x
    
    