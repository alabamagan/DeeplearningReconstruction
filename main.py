# import SimpleITK as sitk
import network
import torch
from torch.autograd import Variable
import torchvision.utils as utils
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import gc
from dataloader import BatchLoader

def train(net, b, trainsteps, epoch=-1):
    if (os.path.isfile("network_E%03d"%epoch) and epoch != -1):
        net = torch.load("network_E%03d"%epoch)
    elif (os.path.isfile("checkpoint_E%03d"%epoch)):
        net = torch.load("checkpoint_E%03d"%epoch)
    else:
        net.train(True)

    optimizer = torch.optim.SGD([{'params': net.convsModules.parameters(),
                                  'lr': 20, 'momentum':1e-1, 'dampening':1e-3},
                                 {'params': net.deconvsModules.parameters(), 'lr': 10},
                                 {'params': net.fcModules.parameters(), 'lr': 1e-3},
                                 {'params': net.bnModules.parameters(), 'lr': 1e-3},
                                 {'params': net.linear1.parameters(),
                                  'lr': 1e-5, 'momentum':0, 'dampening':1e-5}
                                 ])

    fig = plt.figure(1, figsize=[13,6])
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    criterion = torch.nn.SmoothL1Loss().cuda()
    criterion.size_average = False
    normalize = torch.nn.SmoothL1Loss().cuda()
    normalize.size_average = False
    losslist = []
    for i in xrange(trainsteps):
        sample = b[5]
        i2 = sample['032']
        i3 = sample['064']
        gt = sample['ori']
        s = gt.shape
        gt = Variable(torch.from_numpy(gt)).float().cuda()
        i2 = Variable(torch.from_numpy(i2))
        i3 = Variable(torch.from_numpy(i3))

        bstart = np.random.randint(0, gt.data.size()[0] - 10)
        bstop = bstart + 10
        output = net.forward(i2[bstart:bstop].cuda(), i3[bstart:bstop].cuda())
        loss = criterion((output.squeeze()), (gt[bstart:bstop])) / \
               normalize(i3[bstart:bstop].squeeze().float().cuda(), gt[bstart:bstop])
        print "[Step %03d] Loss: %.05f"%(i, loss.data[0])
        losslist.append(loss.data[0])
        loss.backward()
        optimizer.step()
        # #
        # for j in xrange(output.data.size(0) - 1):
        #     ax1.cla()
        #     ax2.cla()
        #     ax3.cla()
        #     ax1.imshow(output.squeeze().cpu().data.numpy()[j], vmin = -1000, vmax=200, cmap="Greys_r")
        #     ax2.imshow(i3.squeeze().cpu().data.numpy()[bstart + j]
        #                - output.squeeze().cpu().data.numpy()[j], vmin = -5, vmax = 5, cmap="jet")
        #     ax3.imshow(i3.squeeze().cpu().data.numpy()[bstart + j],vmin = -1000, vmax=200, cmap="Greys_r")
        #     plt.ion()
        #     plt.draw()
        #     plt.pause(0.01)

        if (i % 100 == 0):
            torch.save(net, "checkpoint_E%03d"%(epoch + 1))

        # Free some meory
        del sample, i2, i3, gt
        gc.collect()

    losslist = np.array(losslist)
    print "======================= End train epoch %03d ======================="%(epoch + 1)
    print "average loss: ", losslist.mean()
    print "final loss: ", losslist[-1]

    torch.save(net, "network_E%03d"%(epoch + 1))

    plt.ioff()
    fig2 = plt.figure(2)
    ax1 = fig2.add_subplot(111)
    ax1.plot(losslist)
    plt.show()

    return losslist

def evalNet(net, b, plot=True):
    if (plot):
        fig = plt.figure(1, figsize=[13,6])
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)

    net.train(False)
    criterion = torch.nn.MSELoss().cuda()
    losslist = []
    for i in xrange(len(b)):


        # Free some meory
        del sample, i2, i3, gt
        gc.collect()

    losslist = np.array(losslist)
    print "======================= End Eval ======================="
    print "average loss: ", losslist.mean()

def main():
    b = BatchLoader("/media/storage/Data/CTReconstruction/LCTSC/Output")

    #=========================
    # Train
    #---------------------
    net = network.Net()
    net.cuda()
    l = train(net, b, trainsteps=2500, epoch=0)
    plt.plot(l)
    plt.show()

    #==================================
    # Evaluation
    #---------------------------
    # netfile = "network_E003"
    # if (os.path.isfile(netfile)):
    #     net = torch.load(netfile)
    #     evalNet(net, b, True)

    pass

if __name__ == '__main__':

    main()