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
        self.channelsize2 = 36

        self.linear1 = nn.Linear(1, 1)

        self.convsModules = nn.ModuleList()
        self.psModules = nn.ModuleList()
        self.fcModules = nn.ModuleList()
        self.bnModules = nn.ModuleList()
        self.poolingLayers = nn.ModuleList()
        self.linearModules = nn.ModuleList()

        self.batchWeightFactor = nn.Linear(1, 1, bias=False)
        self.linearModules['unique'] = self.batchWeightFactor

        self.bnModules['init'] = torch.nn.BatchNorm2d(1)

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
        x = self.bnModules['init'].cuda()(x.unsqueeze(1)).squeeze()
        # x = self.linear1(x.view(np.prod(x.data.size()), 1))
        # print x.data.size()
        # x = x.view_as(x2)


        x = self.im2col(x)
        s = x.data.size()

        #================================================
        # Re-order columes into different types
        # 1. Background
        # 2. Tissues/bench
        # 3. Air
        #-----------------------------------------
        background, TB, Air = None, None, None
        coordBG, coordTB, coordAir =[], [], []
        for i in xrange(s[3]):
            for j in xrange(s[4]):
                for batchNum in xrange(s[0]):
                    patchcoord = [batchNum, i, j] # For recovering patch

                    im = x[batchNum, :, :, i, j]
                    im = im.unsqueeze(0)
                    if (torch.abs(torch.sum(im)) <= 1e-5 ):
                        if (background == None):
                            background = im
                        else:
                            background = torch.cat([background, im] , 0)
                    elif (torch.mode(im) )


        V = None
        for i in xrange(s[3]):
            l_V = None
            for j in xrange(s[4]):


                try:
                    l_conv1 = self.convsModules[0, i, j]
                    l_conv2 = self.convsModules[1, i, j]
                    l_fc1 = self.fcModules[0, i, j]
                    l_fc2 = self.fcModules[1, i, j]
                    l_bn1 = self.bnModules[0, i, j]
                    l_bn2 = self.bnModules[1, i, j]
                    l_bn3 = self.bnModules[2, i, j]
                    l_ps = self.psModules[0, i, j]
                    l_avgPool = self.poolingLayers[0, i, j]

                except KeyError:
                    l_conv1 = nn.Conv2d(1, self.channelsize1, kernel_size=self.kernelsize1,
                                        padding=np.int(self.kernelsize1/2.), bias=False)
                    l_fc1 = nn.ReLU()
                    l_fc2 = nn.ReLU()
                    l_conv2 = nn.Conv2d(self.channelsize1, self.channelsize2, kernel_size=self.kernelsize1,
                                        padding=np.int(self.kernelsize1/2.), bias=False)
                    l_bn1 = nn.BatchNorm2d(self.channelsize1)
                    l_bn2 = nn.BatchNorm2d(self.channelsize2)
                    l_bn3 = nn.BatchNorm2d(1)
                    l_ps = nn.PixelShuffle(np.int(np.floor(np.sqrt(self.channelsize2))))
                    l_avgPool = nn.AvgPool2d(np.int(np.floor(np.sqrt(self.channelsize2))))

                    if (x1.is_cuda):
                        l_conv1.cuda()
                        l_conv2.cuda()
                        l_fc1 = l_fc1.cuda()
                        l_fc2 = l_fc2.cuda()
                        l_bn1 = l_bn1.cuda()
                        l_bn2 = l_bn2.cuda()
                        l_bn3 = l_bn3.cuda()
                        l_ps = l_ps.cuda()
                        l_avgPool = l_avgPool.cuda()


                    self.convsModules[0, i, j] = l_conv1
                    self.convsModules[1, i, j] = l_conv2
                    self.fcModules[0, i, j] = l_fc1
                    self.fcModules[1, i, j] = l_fc2
                    self.bnModules[0, i, j] = l_bn1
                    self.bnModules[1, i, j] = l_bn2
                    self.bnModules[2, i, j] = l_bn3
                    self.psModules[0, i, j] = l_ps
                    self.poolingLayers[0, i, j] = l_avgPool



                l_x = x[:, :, :, i, j].unsqueeze(1)



                if abs(l_x.data.sum()) >= 1e-5
                    ss = l_x.data.size()
                    means = F.avg_pool3d(l_x, kernel_size=[1, ss[-2], ss[-1]]).squeeze()
                    # means = l_x.contiguous().view([ss[0], np.array(list(ss[1::])).prod()]).mean(1).squeeze()
                    weights = self.batchWeightFactor(means.unsqueeze(1)).squeeze()

                    l_x = l_conv1(l_x)
                    l_x = l_bn1(l_x)
                    l_x = l_fc1(l_x)


                    l_x = l_conv2(l_x)
                    l_x = l_bn2(l_x)
                    l_x = l_fc2(l_x)

                    l_x = l_ps(l_x)
                    l_x = l_avgPool(l_x)
                    l_x = l_bn3(l_x)

                    # if expand takes a list as argument, backwards will causes problemm
                    ss = [ss[k] for k in xrange(len(ss) - 1, -1, -1)]
                    temp = weights.expand(ss[0], ss[1], ss[2], ss[3])
                    temp = temp.transpose(0, 2).transpose(1, 3).transpose(0, 1)

                    l_x = l_x * temp
                    l_x = l_x.squeeze()

                    # l_x = l_fc4(l_x)
                else:
                    l_x = l_x.squeeze()

                if (l_V is None):
                    l_V = l_x.unsqueeze(-1).unsqueeze(-1)
                else:
                    try:
                        l_V = torch.cat([l_V, l_x.unsqueeze(-1).unsqueeze(-1)], -1)
                    except RuntimeError:
                        print "[%d,%d]: "%(i,j) + str(l_V.data.size()) + "," + str(l_x.unsqueeze(-1).unsqueeze(-1).data.size())

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

