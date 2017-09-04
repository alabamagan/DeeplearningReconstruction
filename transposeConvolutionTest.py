import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import scipy.misc
import pyinn as P
from torch.autograd import Variable

import matplotlib.pyplot as plt

def main():
    window = np.array([150, 150])
    overlap = np.array([75, 75])
    lena = scipy.misc.face(True)
    print lena.shape
    # lena = torch.from_numpy(lena.transpose(2, 0, 1))
    lena = torch.from_numpy(lena).unsqueeze(0)
    lena = Variable(lena.float()).cuda()

    ##################################################
    # Unfold and convolution
    #-------------------------------------------------
    # lena = lena.unfold(0, window[0], window[0] - overlap[0]).unfold(1, window[1], window[0] - overlap[1])
    # s = lena.size()
    # lena = lena.contiguous().view(s[0]*s[1], 1, window[0],window[1])
    # mask = make_identifier_mask(s[0], s[1])
    # mask = torch.from_numpy(mask.transpose(2, 0, 1)).unsqueeze(1)
    # print mask.size()
    # print lena.size()
    # plt.ion()
    # for i in xrange(s[0]*s[1]):
    #     t = F.conv_transpose2d(Variable(mask[i].unsqueeze(0)).float(), Variable(lena[i].unsqueeze(0)).float())
    #     plt.imshow(t.data.squeeze().numpy())
    #     plt.draw()
    #     plt.pause(0.3)

    ##################################################
    # pyinn
    #-------------------------------------------------
    lena = P.im2col(lena, window, window-overlap, [0, 0]) # (768 x 1024) -> (1 x 150 x 150 x 7 x 9)
    s = lena.data.size()
    plot = torchvision.utils.make_grid(lena.squeeze()
                                       .transpose(0, 2)
                                       .transpose(1, 3)
                                       .contiguous()
                                       .view(s[-1]*s[-2], 1, 150, 150).data
                                       , nrow = 5, padding=10, normalize=True)
    plt.imshow(plot.cpu().numpy().transpose(1,2,0))
    plt.show()


    lena = P.col2im(lena, window, window-overlap, [0, 0]) # (1 x 750 x 950)
    plt.imshow(lena.cpu().data.squeeze().numpy())
    plt.show()


    pass


def make_identifier_mask(xdim, ydim):
    x = np.arange(xdim)
    y = np.arange(ydim)
    zdim = xdim*ydim
    out = np.zeros([xdim, ydim, zdim])
    counter = 0
    for xx in x:
        for yy in y:
            out[xx, yy, counter] = 1
            counter += 1
    return out

if __name__ == '__main__':
    main()