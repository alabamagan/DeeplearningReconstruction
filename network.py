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
        self.convAir1 = nn.Conv2d(1, self.channelsize1, kernel_size=self.kernelsize1,
                                        padding=np.int(self.kernelsize1/2.), bias=False)
        self.convAir2 = nn.Conv2d(self.channelsize1, self.channelsize2, kernel_size=self.kernelsize1,
                                        padding=np.int(self.kernelsize1/2.), bias=False)
        self.convTB1 = nn.Conv2d(1, self.channelsize1, kernel_size=self.kernelsize1,
                                        padding=np.int(self.kernelsize1/2.), bias=False)
        self.convTB2 = nn.Conv2d(self.channelsize1, self.channelsize2, kernel_size=self.kernelsize1,
                                        padding=np.int(self.kernelsize1/2.), bias=False)
        self.bnAir1 = nn.BatchNorm2d(self.channelsize1)
        self.bnAir2 = nn.BatchNorm2d(self.channelsize2)
        self.bnAir3 = nn.BatchNorm2d(1)
        self.bnTB1 = nn.BatchNorm2d(self.channelsize1)
        self.bnTB2 = nn.BatchNorm2d(self.channelsize2)
        self.bnTB3 = nn.BatchNorm2d(1)

        [self.convsModules.append(m) for m in [self.convAir1,
                                               self.convAir2,
                                               self.convTB1,
                                               self.convTB2]]
        [self.bnModules.append(m) for m in [self.bnAir1,
                                            self.bnAir2,
                                            self.bnAir3,
                                            self.bnTB1,
                                            self.bnTB2,
                                            self.bnTB3]]

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


        debugplot = False
        windows = self.windows
        overlap = self.overlap
        inshape = x1.data.size()

        x = x2 - x1
        x = self.bnModules['init'].cuda()(x.unsqueeze(1)).squeeze()
        # x = self.linear1(x.view(np.prod(x.data.size()), 1))
        # print x.data.size()
        # x = x.view_as(x2)

        xx = self.im2col(x2)
        x = self.im2col(x)
        s = x.data.size()

        #================================================
        # Re-order columes into different types
        # 1. Background
        # 2. Tissues/bench
        # 3. Air
        #-----------------------------------------
        background, TB, Air = None, None, None
        coords = {}
        # coordBG, coordTB, coordAir = [], [], []
        for i in xrange(s[3]):
            for j in xrange(s[4]):
                for batchNum in xrange(s[0]):
                    patchcoord = (batchNum, i, j) # For recovering patch

                    imdiff = x[batchNum, :, :, i, j]
                    im = xx[batchNum, :, :, i, j]
                    imdiff = imdiff.unsqueeze(0)
                    if (torch.abs(torch.sum(im)).data[0] <= 1e-4 ):
                        if (background is None):
                            background = imdiff
                        else:
                            background = torch.cat([background, imdiff] , 0)
                        coords[patchcoord] = ['background', background.data.size()[0] - 1]
                    elif (torch.abs(torch.mean(im) + 1000).data[0] < 70):
                        if (Air is None):
                            Air = imdiff
                        else:
                            Air = torch.cat([Air, imdiff], 0)
                        coords[patchcoord] = ['Air', Air.data.size()[0] - 1]
                    else:
                        if (TB is None):
                            TB = imdiff
                        else:
                            TB = torch.cat([TB, imdiff], 0)
                        coords[patchcoord] = ['TB', TB.data.size()[0] - 1]

        Air = Air.unsqueeze(1)
        TB = TB.unsqueeze(1)
        background = background.unsqueeze(1)


        #===================================================
        # Convolution network
        #===================================================
        # Air
        #------- ------------
        # meanAir = F.avg_pool2d(Air, self.windows).squeeze()
        # meanAir = self.linearAir(meanAir.unsqueeze(1)).squeeze()
        # meanAir = F.elu(meanAir)


        Air = self.bnAir1(self.convAir1(Air))
        Air = F.relu(Air)

        Air = self.bnAir2(self.convAir2(Air))
        Air = F.relu(Air)

        Air = F.pixel_shuffle(Air, np.int(np.floor(np.sqrt(self.channelsize2))))
        Air = F.max_pool2d(Air, np.int(np.floor(np.sqrt(self.channelsize2))))

        Air = self.bnAir3(Air) * torch.abs(self.contrastWeightAir.expand_as(Air))
        # sAir = Air.data.size()
        # sAir = [sAir[k] for k in xrange(len(sAir) - 1, -1, -1)]
        # meanAir = meanAir.expand(sAir[0], sAir[1], sAir[2], sAir[3])
        # meanAir = meanAir.transpose(0, 2).transpose(1, 3).transpose(0, 1)
        #
        # Air = Air*meanAir


        # TB
        #---------------------
        # meanTB = F.avg_pool2d(TB, self.windows).squeeze()
        # meanTB = self.linearTB(meanTB.unsqueeze(1)).squeeze()
        # meanTB = F.elu(meanTB)

        TB = self.bnTB1(self.convTB1(TB))
        TB = F.relu(TB)

        TB = self.bnTB2(self.convTB2(TB))
        TB = F.relu(TB)

        TB = F.pixel_shuffle(TB, np.int(np.floor(np.sqrt(self.channelsize2))))
        TB = F.max_pool2d(TB, np.int(np.floor(np.sqrt(self.channelsize2))))

        TB = self.bnTB3(TB) * torch.abs(self.contrastWeightTB.expand_as(TB))
        # sTB = TB.data.size()
        # sTB = [sTB[k] for k in xrange(len(sTB) - 1, -1, -1)]
        # meanTB = meanTB.expand(sTB[0], sTB[1], sTB[2], sTB[3])
        # meanTB = meanTB.transpose(0, 2).transpose(1, 3).transpose(0, 1)
        # TB = TB*meanTB

        #===================================================
        # Reconstruct back to image colume
        #-----------------------------------
        v = None
        for i in xrange(s[3]):
            l_v = None
            for j in xrange(s[4]):
                l_l_v = None
                for batchNum in xrange(s[0]):
                    patchcoord = (batchNum, i, j)

                    pair = coords[patchcoord]
                    if pair[0] == 'background':
                        im = background[pair[1]]
                    elif pair[0] == 'Air':
                        im = Air[pair[1]]
                    elif pair[0] == 'TB':
                        im = TB[pair[1]]
                    else:
                        print "Something wrong with patch: ", patchcoord
                        im = None

                    im = im.unsqueeze(-1).unsqueeze(-1)

                    if l_l_v is None:
                        l_l_v = im
                    else:
                        l_l_v = torch.cat([l_l_v, im], 0)
                    # ==== End For ====
                if (l_v is None):
                    l_v = l_l_v
                else:
                    l_v = torch.cat([l_v, l_l_v], 4)
                # ==== End For ====
            if (v is None):
                v = l_v
            else:
                v = torch.cat([v, l_v], 3)
            # ==== End For ====

        x = self.col2im(v)
        # x = F.elu(x)

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

