import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from algorithm import ExtractPatchIndexs
from pyinn import im2col, col2im

# testing
import matplotlib.pyplot as plt
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        # self.windows = np.array([2, 32, 32])
        # self.overlap = np.array([0, 16, 16])

        self.conv1 = nn.Conv2d(1, 24, (5, 5))
        self.conv2 = nn.Conv2d(24, 60, 5)
        self.deconv1 = nn.ConvTranspose2d(24, 1, (5, 5))
        self.deconv2 = nn.ConvTranspose2d(60, 24, 5)

        self.fc1 = nn.ELU()
        self.fc2 = nn.ELU()
        self.fc3 = nn.ELU()

        self.linear1 = nn.Linear(1, 1, bias=False)

        self.convsModules = nn.ModuleList()
        self.convsModules.append(self.conv1)
        self.convsModules.append(self.conv2)
        self.convsModules.append(self.deconv1)
        self.convsModules.append(self.deconv2)

        self.miscModules = nn.ModuleList()
        self.miscModules.append(self.fc1)
        self.miscModules.append(self.fc2)
        self.miscModules.append(self.fc3)

        self.windows = np.array([2, 16, 16])
        self.overlap = np.array([0, 8, 8])

        self.norm1 = nn.BatchNorm2d(24)
        self.norm2 = nn.BatchNorm2d(60)

    def forward(self, x1, x2):
        """
        x2 should have better resolution
        :param x1:
        :param x2:
        :return:
        """
        assert x1.is_cuda and x2.is_cuda, "Inputs are not in GPU!"


        debugplot = False
        windows = self.windows
        overlap = self.overlap
        inshape = x1.data.size()

        x = x2 - x1

        # Unfold
        #-----------
        x.data = x.data.unfold(3, windows[1], overlap[1]).unfold(4, windows[2], overlap[2]).squeeze()
        s = x.data.size()
        x = x.contiguous().view(1, s[0]*s[1], windows[1], windows[2])
        x = x.transpose(0, 1)
        outX = x.contiguous().view(s[0]*s[1], 1, windows[1], windows[2])

        x = self.conv1(outX)
        x = self.norm1(x)
        x = self.fc1(x)

        x = self.conv2(torch.squeeze(x))
        x = self.norm2(x)
        x = self.fc2(x)

        x = self.deconv2(x)
        x = self.fc3(x)

        s2 = x.data.size()
        x = self.deconv1(x.view(s2[0], s2[1], s2[2], s2[3]))

        V = None
        for i in xrange(s[0]):
            l_v = None
            for j in xrange(s[1]):
                counter = i*s[1] + j

                l_x = x[counter].unsqueeze(0).unsqueeze(0)
                if (l_v is None):
                    l_v = l_x
                else:
                    l_v = torch.cat([l_v, l_x], 0)

            if (V is None):
                V = l_v
            else:
                V = torch.cat([V, l_v], 1)

        V = V.squeeze()
        V = V.transpose(0, 3).transpose(1, 2).unsqueeze(0) # (1 x win1 x win2 x p1 x p2)

        outX2 = col2im(V.contiguous() # remember to contiguous() here
                       , windows[1:], windows[1:] - overlap[1:], [0,0])

        x2s = outX2.data.size()
        outX2 = self.linear1(outX2.view(np.prod(x2s), 1))
        outX2 = outX2.view_as(x2)
        x = x2 - outX2
        # x = outX2
        return x


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

