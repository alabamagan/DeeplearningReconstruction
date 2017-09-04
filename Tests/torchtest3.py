import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

def main():
    im64 = np.load("./TestData/064.npy")
    im64.transpose(2, 0, 1)
    print im64.shape

    im64 = Variable(torch.from_numpy(im64.reshape(1, 1, 168, 512, 512)))
    im64 = im64.cuda(0)

    fc1 = nn.Conv3d(1, 10, (3, 64, 64))
    fc1 = fc1.cuda(0)
    out = fc1(im64)
    s = out.data.size()
    print s

    # imglist = [out.data[0,i].view(1, s[2], s[3]) for i in xrange(out.size(1))]
    # # for i in xrange(len(imglist)):
    # #     print imglist[i].numpy().shape
    # #     plt.imsave("./TestData/conv_%02d.png"%i,imglist[i].numpy()[0])
    #
    # plot = torchvision.utils.make_grid(imglist, padding=10, normalize=True)
    # plot = plot.cpu()
    # plot = plot.numpy().transpose(1, 2, 0)
    # plt.imshow(plot)
    # plt.show()

    # print torchvision.utils.save_image(out.data, "./TestData/plot.png")
    pass

#
# def main():
#     im64 = np.load("./TestData/064.npy")
#     im64.transpose(2, 0, 1)
#     print im64.shape
#
#     im64 = Variable(torch.from_numpy(im64[50].reshape(1, 1, 512, 512)))
#     im64 = im64.cuda(0)
#
#     fc1 = nn.Conv2d(1, 10, (64, 64))
#     fc1 = fc1.cuda(0)
#     out = fc1(im64)
#     s = out.data.size()
#
#     imglist = [out.data[0,i].view(1, s[2], s[3]) for i in xrange(out.size(1))]
#     # for i in xrange(len(imglist)):
#     #     print imglist[i].numpy().shape
#     #     plt.imsave("./TestData/conv_%02d.png"%i,imglist[i].numpy()[0])
#
#     plot = torchvision.utils.make_grid(imglist, padding=10, normalize=True)
#     plot = plot.cpu()
#     plot = plot.numpy().transpose(1, 2, 0)
#     plt.imshow(plot)
#     plt.show()
#
#     # print torchvision.utils.save_image(out.data, "./TestData/plot.png")
#     pass

if __name__ == '__main__':
    main()