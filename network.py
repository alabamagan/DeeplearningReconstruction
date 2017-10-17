import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from algorithm import ExtractPatchIndexs
from pyinn import im2col, col2im
from pyinn.im2col import Im2Col, Col2Im
import visdom
# testing
import matplotlib.pyplot as plt
import numpy as np

vis = visdom.Visdom(port=80)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.kernelsize1 = 9
        self.kernelsize2 = 5
        self.channelsize1 =8
        self.channelsize2 = 49

        self.windows = np.array([32, 32])
        self.overlap = np.array([16, 16])

        self.convsModules = nn.ModuleList()
        self.psModules = nn.ModuleList()
        self.fcModules = nn.ModuleList()
        self.bnModules = nn.ModuleList()
        self.poolingLayers = nn.ModuleList()
        self.linearModules = nn.ModuleList()
        self.miscParams = nn.ParameterList()
        self.bnModules['init'] = nn.BatchNorm2d(1)

        #===============================
        # Patch sorting kernel
        #-----------------------
        self.numOfTypes = 3
        self.convSort1 = torch.nn.Conv2d(1, 6, 5)
        self.convSort2 = torch.nn.Conv2d(6, 15, 5)
        self.poolSort1 = torch.nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(np.prod(((self.windows - 4)/2 - 4 )/2) * 15, 28)
        self.fc2 = nn.Linear(28, self.numOfTypes)
        self.convsModules.extend([self.convSort1, self.convSort2])
        self.fcModules.extend([self.fc1, self.fc2, self.poolSort1])

        self.ConvNetwork = []
        for i in xrange(self.numOfTypes):
            sub_net = {}
            sub_net['conv1'] = nn.Conv2d(1,
                                         self.channelsize1,
                                         kernel_size=self.kernelsize1,
                                         padding=np.int(self.kernelsize1/2.),
                                         bias=False)

            sub_net['conv2'] = nn.Conv2d(self.channelsize1,
                                         self.channelsize2,
                                         kernel_size=self.kernelsize1,
                                         padding=np.int(self.kernelsize1/2.),
                                         bias=False)
            sub_net['bn1'] = nn.BatchNorm2d(self.channelsize1)
            sub_net['bn2'] = nn.BatchNorm2d(self.channelsize2)
            sub_net['bn3'] = nn.BatchNorm2d(1)
            sub_net['linear'] = nn.Parameter(torch.ones(1), requires_grad=True)
            self.convsModules.extend([sub_net['conv1'], sub_net['conv2']])
            self.bnModules.extend([sub_net['bn1'], sub_net['bn2'], sub_net['bn3']])
            self.miscParams.append(sub_net['linear'])
            self.ConvNetwork.append(sub_net)


        self.current_step = 0
        self.current_epoch = 0
        self.loss_list = []

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

        x = x2 - x1
        x = self.bnModules['init'].cuda()(x.unsqueeze(1)).squeeze()
        # x = self.linear1(x.view(np.prod(x.data.size()), 1))
        # print x.data.size()
        # x = x.view_as(x2)

        xx = self.im2col(x)
        x = self.im2col(x)
        s = x.data.size()

        #============================================
        # Patch sorting layer
        #----------------------------------------
        # Stack xx into one column for sorting
        xx = xx.transpose(1, 3).transpose(2, 4).contiguous()
        xxs = xx.data.size()
        xx = xx.view(xxs[0]*xxs[1]*xxs[2], 1, xxs[3], xxs[4])

        xx = self.poolSort1(F.relu(self.convSort1(xx))) # (B, C, 14, 14)
        xx = self.poolSort1(F.relu(self.convSort2(xx))) # (B, C, 5, 5)
        xx = xx.view(-1, np.prod(((self.windows - 4)/2 - 4 )/2) * 15)
        xx = F.relu(self.fc1(xx))
        xx = F.relu(self.fc2(xx))
        xx = xx.view(xxs[0], xxs[1], xxs[2], self.numOfTypes)

        coords = {}
        sortedcols = [None for i in xrange(self.numOfTypes)]
        for i in xrange(s[3]):
            for j in xrange(s[4]):
                for batchNum in xrange(s[0]):
                    patchcoord = (batchNum, i, j)
                    imdiff = x[batchNum, :, :, i, j]
                    imdiff = imdiff.unsqueeze(0)
                    value, index = xx[batchNum,i, j].data.max(0)
                    index = index[0]

                    if sortedcols[index] is None:
                        sortedcols[index] = imdiff
                    else:
                        sortedcols[index] = torch.cat([sortedcols[index], imdiff], 0)
                    coords[patchcoord] = [index, sortedcols[index].data.size()[0] - 1]


        for i in xrange(len(sortedcols)):
            col = sortedcols[i]
            if col is None:
                print "Col ", i, " is None"
                continue

            displayrangeIm = [-1000, 300]
            displayrangeDiff = [-15, 15]
            normIm = lambda inIm: inIm.clip(displayrangeIm[0], displayrangeIm[1])/\
                                  float(displayrangeIm[1] - displayrangeIm[0])
            normDiff = lambda inIm: inIm.clip(displayrangeDiff[0], displayrangeDiff[1])/\
                                  float(displayrangeDiff[1] - displayrangeDiff[0])
            d = normDiff(col[0:200].unsqueeze(1).data.cpu().numpy())
            d += d.min()
            vis.images(d, nrow=20, win="SortedColume%i"%i, env="Plots")
            col = col.unsqueeze(1).contiguous()
            conv1, conv2 = [self.ConvNetwork[i]['conv1'], self.ConvNetwork[i]['conv2']]
            bn1, bn2, bn3 = [self.ConvNetwork[i]['bn1'], self.ConvNetwork[i]['bn2'], self.ConvNetwork[i]['bn3']]
            mean = self.ConvNetwork[i]['linear']
            col = F.relu(bn1(conv1(col)))
            col = F.relu(bn2(conv2(col)))
            col = F.pixel_shuffle(col, np.int(np.floor(np.sqrt(self.channelsize2))))
            col = F.avg_pool2d(col, np.int(np.floor(np.sqrt(self.channelsize2))))
            col = bn3(col)
            col = col * torch.abs(mean.expand_as(col))
            sortedcols[i] = col




        v = None
        for i in xrange(s[3]):
            l_v = None
            for j in xrange(s[4]):
                l_l_v = None
                for batchNum in xrange(s[0]):
                    patchcoord = (batchNum, i, j)

                    pair = coords[patchcoord]
                    im = sortedcols[pair[0]][pair[1]]
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


        x = self.col2im(v)
        s = x.data.size()
        # x = F.elu(x)

        # Zero out some values
        for i in xrange(s[-2]):
            for j in xrange(s[-1]):
                if (i - s[-2]/2.) ** 2 + (j - s[-1]/2.)** 2 > ((s[-2]/2.) ** 2. + (s[-1]/2.) ** 2.)/2. + 1:
                    x[:,i, j] = 0

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

