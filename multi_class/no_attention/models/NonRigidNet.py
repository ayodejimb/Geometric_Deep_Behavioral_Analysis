from layers.rigidtransform import RigidTransform
from layers.nonrigidtransform import NonRigidTransform
from layers.rigidtransforminit import RigidTransformInit
from layers.nonrigidtransforminit import NonRigidTransformInit
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchgeometry.core as tgmc

depth_1 = 32
kernel_size_1 = 3
stride_size = 2
depth_2 = 128
kernel_size_2 = 1
num_hidden = 32
num_labels = 3
dims = 2

class NonRigidNet(nn.Module):
    def __init__(self, mod = 'NonRigidTransform', num_frames = 7000, num_joints = 8, param=110, run=0):
        super(NonRigidNet, self).__init__()
        self.num_channels = num_joints * dims
        self.mod = mod
        self.num_frames = num_frames
        self.num_joints = num_joints
        if mod == 'NonRigidTransform':
            self.rot = NonRigidTransform(num_frames,num_joints, run)
        elif mod == 'NonRigidTransformInit':
            self.rot = NonRigidTransformInit(num_frames,num_joints, run)

        self.interm_conv = nn.Sequential(
            nn.Conv1d(self.num_channels, depth_2, kernel_size=kernel_size_1, stride=stride_size),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.LSTM = nn.LSTM(param, hidden_size=12, bidirectional =True)    
        self.pool=nn.MaxPool1d(kernel_size=2, stride=stride_size)
        self.fc1 = nn.Sequential(
            nn.Linear(depth_2*24, num_hidden),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(num_hidden, num_labels),
        )
        
    def forward(self, x):
        x = x.view(x.size(0),self.num_frames*self.num_joints,1,dims)
        x = self.rot(x)
        x = x.view(x.size(0),self.num_joints*dims,self.num_frames)   
        x = self.pool(self.interm_conv(x))
        x,  _= self.LSTM(x)
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
