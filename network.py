import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from algorithm import ExtractPatchIndexs
from pyinn import im2col, col2im
from pyinn.im2col import Im2Col, Col2Im

# testing
import matplotlib.pyplot as plt
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.kernelsize1 = 15
        self.kernelsize2 = 5
        self.channelsize1 = 12
        self.channelsize2 = 24

        self.linear1 = nn.Linear(1, 1)

        self.convsModules = nn.ModuleList()
        self.deconvsModules = nn.ModuleList()
        self.fcModules = nn.ModuleList()
        self.bnModules = nn.ModuleList()

        self.windows = np.array([32, 32])
        self.overlap = np.array([16, 16])


        self.im2col = Im2Col(self.windows, self.windows - self.overlap, 0)
        self.col2im = Col2Im(self.windows, self.windows - self.overlap, 0)

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

        self.bnModules['init'] = torch.nn.BatchNorm2d(1)
        x = torch.nn.BatchNorm2d(1).cuda()(x.unsqueeze(1)).squeeze()
        # x = self.linear1(x.view(np.prod(x.data.size()), 1))
        # print x.data.size()
        # x = x.view_as(x2)


        x = self.im2col(x)
        s = x.data.size()

        V = None
        for i in xrange(s[3]):
            l_V = None
            for j in xrange(s[4]):
                try:
                    l_conv1 = self.convsModules[0, i, j]
                    l_conv2 = self.convsModules[1, i, j]
                    l_fc1 = self.fcModules[0, i, j]
                    l_fc2 = self.fcModules[1, i, j]
                    l_fc3 = self.fcModules[2, i, j]
                    # l_fc4 = self.fcModules[3, i, j]
                    l_bn1 = self.bnModules[0, i, j]
                    l_bn2 = self.bnModules[1, i, j]
                    # l_bn3 = self.bnModules[3, i, j]
                    l_deconv1 = self.deconvsModules[0, i, j]
                    l_deconv2 = self.deconvsModules[1, i, j]
                except KeyError:
                    l_conv1 = nn.Conv2d(1, self.channelsize1, kernel_size=self.kernelsize1)
                    l_conv2 = nn.Conv2d(self.channelsize1, self.channelsize2, kernel_size=self.kernelsize2)
                    l_fc1 = nn.ELU()
                    l_fc2 = nn.ELU()
                    l_fc3 = nn.ELU()
                    # l_fc4 = nn.ELU()
                    l_bn1 = nn.BatchNorm2d(self.channelsize1)
                    l_bn2 = nn.BatchNorm2d(self.channelsize2)
                    # l_bn3 = nn.BatchNorm2d(24)
                    l_deconv1 = nn.ConvTranspose2d(self.channelsize1, 1, kernel_size=self.kernelsize1)
                    l_deconv2 = nn.ConvTranspose2d(self.channelsize2, self.channelsize1, kernel_size=self.kernelsize2)
                    l_conv1.train()
                    if (x1.is_cuda):
                        l_conv1.cuda()
                        l_conv2.cuda()
                        l_fc1.cuda()
                        l_fc2.cuda()
                        l_fc3.cuda()
                        # l_fc4.cuda()
                        l_bn1.cuda()
                        l_bn2.cuda()
                        # l_bn3.cuda()
                        l_deconv1.cuda()
                        l_deconv2.cuda()
                    self.convsModules[0, i, j] = l_conv1
                    self.convsModules[1, i, j] = l_conv2
                    self.fcModules[0, i, j] = l_fc1
                    self.fcModules[1, i, j] = l_fc2
                    self.fcModules[2, i, j] = l_fc3
                    # self.fcModules[3, i, j] = l_fc4
                    self.bnModules[0, i, j] = l_bn1
                    self.bnModules[1, i, j] = l_bn2
                    # self.bnModules[2, i, j] = l_bn3
                    self.deconvsModules[0, i, j] = l_deconv1
                    self.deconvsModules[1, i, j] = l_deconv2



                l_x = x[:, :, :, i, j].unsqueeze(1)

                if abs(l_x.data.sum()) >= 1e-5:
                    l_x = l_conv1(l_x)
                    l_x = l_bn1(l_x)
                    l_x = l_fc1(l_x)

                    l_x = l_conv2(l_x)
                    l_x = l_bn2(l_x)
                    l_x = l_fc2(l_x)

                    l_x = l_deconv2(l_x)
                    # l_x = l_bn3(l_x)
                    l_x = l_fc3(l_x)

                    l_x = l_deconv1(l_x).squeeze()
                    # l_x = l_fc4(l_x)
                else:
                    l_x = l_x.squeeze()

                if (l_V is None):
                    l_V = l_x.unsqueeze(-1).unsqueeze(-1)
                else:
                    l_V = torch.cat([l_V, l_x.unsqueeze(-1).unsqueeze(-1)], -1)

            if (V is None):
                V = l_V
            else:
                V = torch.cat([V, l_V], -2)


        x = self.col2im(V)

        ## Bad performance
        # x2s = x.data.size()
        # x = self.linear1(x.view(np.prod(x2s), 1))
        # x = x.view_as(x2)
        x = x2 - x
        return x


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

