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

        self.kernelsize1 = 11
        self.kernelsize2 = 5

        self.linear1 = nn.Linear(1, 1, bias=False)

        self.convsModules = nn.ModuleList()
        self.deconvsModules = nn.ModuleList()
        self.fcModules = nn.ModuleList()
        self.bnModules = nn.ModuleList()

        self.windows = np.array([16, 16])
        self.overlap = np.array([8, 8])

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

        x = im2col(x, self.windows, self.windows - self.overlap, 0)
        s = x.data.size()

        V = None
        for i in xrange(s[1]):
            l_V = None
            for j in xrange(s[2]):
                try:
                    l_conv1 = self.convsModules[0, i, j]
                    l_conv2 = self.convsModules[1, i, j]
                    l_fc1 = self.fcModules[0, i, j]
                    l_fc2 = self.fcModules[1, i, j]
                    l_fc3 = self.fcModules[2, i, j]
                    l_bn1 = self.bnModules[0, i, j]
                    l_bn2 = self.bnModules[1, i, j]
                    l_deconv1 = self.deconvsModules[0, i, j]
                    l_deconv2 = self.deconvsModules[1, i, j]
                except KeyError:
                    l_conv1 = nn.Conv2d(1, 24, kernel_size=self.kernelsize1)
                    l_conv2 = nn.Conv2d(24, 60, kernel_size=self.kernelsize2)
                    l_fc1 = nn.ELU()
                    l_fc1.train(false)
                    l_fc2 = nn.ELU()
                    l_fc3 = nn.ELU()
                    l_bn1 = nn.BatchNorm2d(24)
                    l_bn2 = nn.BatchNorm2d(60)
                    l_deconv1 = nn.ConvTranspose2d(24, 1, kernel_size=self.kernelsize1)
                    l_deconv2 = nn.ConvTranspose2d(60, 24, kernel_size=self.kernelsize2)
                    if (x1.is_cuda):
                        l_conv1.cuda()
                        l_conv2.cuda()
                        l_fc1.cuda()
                        l_fc2.cuda()
                        l_fc3.cuda()
                        l_bn1.cuda()
                        l_bn2.cuda()
                        l_deconv1.cuda()
                        l_deconv2.cuda()
                    self.convsModules[0, i, j] = l_conv1
                    self.convsModules[1, i, j] = l_conv2
                    self.fcModules[0, i, j] = l_fc1
                    self.fcModules[1, i, j] = l_fc2
                    self.fcModules[2, i, j] = l_fc3
                    self.bnModules[0, i, j] = l_bn1
                    self.bnModules[1, i, j] = l_bn2
                    self.deconvsModules[0, i, j] = l_deconv1
                    self.deconvsModules[1, i, j] = l_deconv2

                l_x = x[:,i, j].unsqueeze(1)
                l_x = l_conv1(l_x)
                l_x = l_bn1(l_x)
                l_x = l_fc1(l_x)

                l_x = l_conv2(l_x)
                l_x = l_bn2(l_x)
                l_x = l_fc2(l_x)

                l_x = l_deconv2(l_x)
                l_x = l_fc3(l_x)

                l_x = l_deconv1(l_x)

                if (l_V is None):
                    l_V = l_x.unsqueeze(2)
                else:
                    l_V = torch.cat([l_x.unsqueeze(2), l_V], 2)

            if (V is None):
                V = l_V
            else:
                V = torch.cat([l_V, V], 1)

        x = col2im(V.contiguous(), self.windows, self.windows - self.overlap, 0)
        # V = None
        # for i in xrange(s[0]):
        #     l_v = None
        #     for j in xrange(s[1]):
        #         counter = i*s[1] + j
        #
        #         l_x = x[counter].unsqueeze(0).unsqueeze(0)
        #         if (l_v is None):
        #             l_v = l_x
        #         else:
        #             l_v = torch.cat([l_v, l_x], 0)
        #
        #     if (V is None):
        #         V = l_v
        #     else:
        #         V = torch.cat([V, l_v], 1)
        #
        # V = V.squeeze()
        # V = V.transpose(0, 3).transpose(1, 2).unsqueeze(0) # (1 x win1 x win2 x p1 x p2)


        # x2s = outX2.data.size()
        # outX2 = self.linear1(outX2.view(np.prod(x2s), 1))
        # outX2 = outX2.view_as(x2)
        x = x2 - x
        # x = outX2
        return x


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

