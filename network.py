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

        self.kernelsize1 = 9
        self.kernelsize2 = 5
        self.channelsize1 = 5
        self.channelsize2 = 25

        self.convsModules = nn.ModuleList()
        self.psModules = nn.ModuleList()
        self.fcModules = nn.ModuleList()
        self.bnModules = nn.ModuleList()
        self.poolingLayers = nn.ModuleList()
        self.linearModules = nn.ModuleList()
        self.miscParams = nn.ParameterList()
        self.contrastWeight = nn.Parameter(torch.ones(1), requires_grad=True)
        self.contrastWeightAir = nn.Parameter(torch.ones(1), requires_grad=True)
        self.contrastWeightTB = nn.Parameter(torch.ones(1), requires_grad=True)
        self.miscParams.append(self.contrastWeight)
        self.miscParams.append(self.contrastWeightAir)
        self.miscParams.append(self.contrastWeightTB)

        #====================================
        # Create modules
        #---------------------
        self.bnModules['init'] = torch.nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, self.channelsize1, kernel_size=self.kernelsize1, bias=False)
        self.conv2 = nn.Conv2d(self.channelsize1, self.channelsize2, kernel_size=self.kernelsize1, bias=False)
        self.deconv1 = nn.ConvTranspose2d(self.channelsize1, 1, kernel_size=self.kernelsize1, bias=False)
        self.deconv2 = nn.ConvTranspose2d(self.channelsize2, self.channelsize1, kernel_size=self.kernelsize1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.channelsize1)
        self.bn2 = nn.BatchNorm2d(self.channelsize2)
        self.bn3 = nn.BatchNorm2d(1)

        [self.convsModules.append(m) for m in [self.conv1,
                                               self.conv2,
                                               self.deconv1,
                                               self.deconv2]]
        [self.bnModules.append(m) for m in [self.bn1,
                                            self.bn2,
                                            self.bn3]]

        self.linear = nn.Linear(1, 1, bias=False)
        self.linearAir = nn.Linear(1, 1, bias=False)
        self.linearTB = nn.Linear(1, 1, bias=False)
        [self.linearModules.append(m) for m in [self.linear, self.linearAir, self.linearTB]]


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


        debugplot = True
        windows = self.windows
        overlap = self.overlap
        inshape = x1.data.size()

        x = x2 - x1
        x = self.bnModules['init'].cuda()(x.unsqueeze(1)).squeeze()
        # print x.data.size()
        # x = x.view_as(x2)

        x = self.im2col(x)
        s = x.data.size()

        x = x.contiguous().transpose(1, 3).transpose(2, 4).contiguous()
        ts = x.data.size()
        x = x.view(ts[0]*ts[1]*ts[2], 1, ts[3], ts[4])

        x = self.bn1(self.conv1(x))
        x = F.relu(x)

        x = self.bn2(self.conv2(x))
        x = F.relu(x)

        x = self.deconv2(x)
        x = self.deconv1(x)
        x = self.bn3(x) * torch.abs(self.contrastWeight.expand_as(x))

        x = x.contiguous().view(ts[0], ts[1], ts[2], ts[3], ts[4]).contiguous()
        x = x.transpose(1, 3).transpose(2, 4).contiguous()

        x = self.col2im(x)
        # if (debugplot):
        #     grid = torchvision.utils.make_grid(x.data.unsqueeze(1), nrow=4, padding=5, normalize=True)
        #     plt.ioff()
        #     plt.imshow(grid.cpu().numpy().transpose(1, 2, 0))
        #     plt.show()
        #     # plt.ion()
        #     # for i in xrange(31):
        #     #     for j in xrange(31):
        #     #         plt.imshow(x.data[0, :, :, i, j].cpu().numpy())
        #     #         plt.draw()
        #     #         plt.pause(0.2)
        # # x = F.elu(x)

        ## Bad performance
        # x2s = x.data.size()
        # x = self.linear(x.view(np.prod(x2s), 1))
        # x = x.view_as(x2)
        # print self.linear.bias
        # x = x * torch.abs(self.contrastWeight.expand_as(x))
        # print self.contrastWeight.data[0]
        x = x2 - x
        return x


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

