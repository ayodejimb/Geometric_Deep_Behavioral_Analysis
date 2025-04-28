import torch 
import torch.nn as nn
import torch.nn.functional as F


class temporal_2(nn.Module):
    def __init__(self, channels, r):
        super(temporal_2, self).__init__()
        self.channels = channels
        self.r = r
        self.output_sig = None
        self.linear = nn.Sequential(
            nn.Linear(in_features=self.channels, out_features=self.channels//self.r, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.channels//self.r, out_features=self.channels, bias=True))

    def forward(self, x):
        max = F.adaptive_max_pool1d(x, output_size=1)
        avg = F.adaptive_avg_pool1d(x, output_size=1)
        linear_max = self.linear(max.view(1, max.shape[0]))
        linear_avg = self.linear(avg.view(1, avg.shape[0]))
        output = linear_max + linear_avg
        self.output_sig = F.relu(output)        
        output_times_x = self.output_sig.view(-1, 1) * x
        return output_times_x