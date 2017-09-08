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
import argparse
import sys
from dataloader import BatchLoader

def train(net, b, trainsteps, epoch=-1):
    net.train()

    optimizer = None
    #
    # fig = plt.figure(1, figsize=[13,6])
    # ax1 = fig.add_subplot(131)
    # ax2 = fig.add_subplot(132)
    # ax3 = fig.add_subplot(133)

    criterion = torch.nn.SmoothL1Loss().cuda()
    criterion.size_average = True
    normalize = torch.nn.SmoothL1Loss().cuda()
    normalize.size_average = True
    losslist = []
    for i in xrange(trainsteps):
        index = np.random.randint(0, 5)
        sample = b[index]
        i2 = sample['032']
        i3 = sample['064']
        gt = sample['ori']
        s = gt.shape
        gt = Variable(torch.from_numpy(gt)).float().cuda()
        i2 = Variable(torch.from_numpy(i2))
        i3 = Variable(torch.from_numpy(i3))

        offset = 10
        bstart = np.random.randint(0, i2.data.size()[0] - offset)
        bstop = bstart + offset

        output = net.forward(i2[bstart:bstop].cuda(), i3[bstart:bstop].cuda())
        #=================================================
        # Add modules to optimizer or else it wont budge
        #-----------------------------------------------
        if (optimizer == None):
            optimizer = torch.optim.SGD([{'params': net.convsModules.parameters(),
                                          'lr': 5, 'momentum':1e-2, 'dampening': 1e-2},
                                         {'params': net.deconvsModules.parameters(), 'lr': 2},
                                         {'params': net.fcModules.parameters(), 'lr': 1e-3},
                                         {'params': net.bnModules.parameters(), 'lr': 1e-3},
                                         {'params': net.linear1.parameters(),
                                          'lr': 1e-1, 'momentum':0, 'dampening':1e-5}
                                         ])
            optimizer.zero_grad()

        loss = criterion((output.squeeze()), (gt[bstart:bstop])) / normalize(i3[bstart:bstop].float().cuda(), gt[bstart:bstop])
        print "[Step %03d] Loss: %.010f  on b[%i]"%(i, loss.data[0], index)
        losslist.append(loss.data[0])
        loss.backward()


        optimizer.step()

        #======================================
        # Plot for visualization of result
        #----------------------------------
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

def main(parserargs):

    #=========================
    # Error check
    #---------------------
    assert os.path.isdir(a.input), "Input directory does not exist!"
    assert a.epoch >= 0, "Epoch must be positive or zero!"
    if (a.usecuda):
        assert torch.has_cudnn, "CUDA is not supported on this machine!"
    if (a.checkpoint):
        assert os.path.isfile(a.checkpoint + "_E%03d"%(a.epoch + 1)), "Check point file not found!"
        net = torch.load(a.checkpoint)
    elif (os.path.isfile("network_E%03d"%(a.epoch + 1))):
        net = torch.load("network_E%03d"%(a.epoch + 1))
    else:
        net = network.Net()



    #=========================
    # Train
    #---------------------
    b = BatchLoader(a.input)
    if a.usecuda:
        net.cuda()
    net.zero_grad()

    if (a.train):
        l = train(net, b, trainsteps=1000, epoch=a.epoch)
        if a.plot:
            plt.plot(l)
            plt.show()
        else:
            fig = plt.figure()
            fig.set_tight_layout(True)
            ax1 = fig.add_subplot(111)
            ax1.set_title("Training network Epoch %03d"%(a.epoch + 1))
            ax1.set_xlabel("Step")
            ax1.set_ylabel("Loss")
            plt.savefig(fig, "fig.png")

    #==================================
    # Evaluation
    #---------------------------
    # netfile = "network_E003"
    # if (os.path.isfile(netfile)):
    #     net = torch.load(netfile)
    #     evalNet(net, b, True)3cfn
    else:
        evalNet(net, b, a.plot)

    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training reconstruction from less projections.")
    parser.add_argument("input", metavar='input', action='store',
                        help="Train/Target input", type=str)
    parser.add_argument("-o", metavar='output', dest='output', action='store', type=str, default=None,
                        help="Set where to store outputs for eval mode")
    parser.add_argument("-p", dest='plot', action='store_true', default=False,
                        help="Select whether to disply the plot for stepwise loss")
    parser.add_argument("-e", "--epoch", dest='epoch', action='store', type=int, default=0,
                        help="Select network epoch.")
    parser.add_argument("--load", dest='checkpoint', action='store', default='',
                        help="Specify network checkpoint.")
    parser.add_argument("--useCUDA", dest='usecuda', action='store_true',default=False,
                        help="Set whether to use CUDA or not.")
    parser.add_argument("--train", dest='train', action='store_true', default=False,
                        help="Set whether to train or evaluate, default is eval")
    parser.add_argument("--train-params", dest='trainparams', action='store', type=dir, default=None,
                        help="Path to a file with dictionary of training parameters written inside")
    a = parser.parse_args()
    print a

    if (not a.train and not a.output):
        print "Error!  Must specify output by -o for evaluation mode!"
        parser.print_help()


    main(a)

    # main()