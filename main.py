#!/home/lwong/Toolkits/Anaconda2/bin/python
#  import SimpleITK as sitk
import network
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import gc
import logging
import argparse
from dataloader import BatchLoader

#============================================
# Prepare global logger
logging.getLogger(__name__).setLevel(10)

def train(net, b, trainsteps, epoch=-1, plot=False, params=None):
    """
    Descriptions
    ------------
      Train the network.

    :param network.Net  net:        Network to be trained
    :param BatchLoader  b:          Batch loader for loading training data
    :param int          trainsteps: Number steps per epoch
    :param int          epoch:      Number of epoch
    :param bool         plot:       Set true to show plot, require X-server screen resources
    :param dict         params:     Not supported yet
    :return:
    """
    net.train()

    optimizer = None

    if (plot):
        fig = plt.figure(1, figsize=[13,6])
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)

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
                                          'lr': 0.02, 'momentum':1e-2, 'dampening': 1e-2},
                                         {'params': net.deconvsModules.parameters(), 
                                          'lr': 0.02, 'momentum':1e-3, 'dampling':1e-2},
                                         {'params': net.fcModules.parameters(), 'lr': 1e-3},
                                         {'params': net.bnModules.parameters(), 'lr': 1e-3},
                                         {'params': net.linear1.parameters(),
                                          'lr': 1e-3, 'momentum':0, 'dampening':1e-5}
                                         ])
            optimizer.zero_grad()

        loss = criterion((output.squeeze()), (gt[bstart:bstop])) / normalize(i3[bstart:bstop].float().cuda(), gt[bstart:bstop])
        print "[Step %03d] Loss: %.010f  on b[%i]"%(i, loss.data[0], index)
        logging.getLogger(__name__).log(20, "[Step %03d] Loss: %.010f  on b[%i]"%(i, loss.data[0], index))
        losslist.append(loss.data[0])
        loss.backward()


        optimizer.step()

        #======================================
        # Plot for visualization of result
        #----------------------------------
        if (plot):
            for j in xrange(output.data.size(0) - 1):
                ax1.cla()
                ax2.cla()
                ax3.cla()
                ax1.imshow(output.squeeze().cpu().data.numpy()[j], vmin = -1000, vmax=200, cmap="Greys_r")
                ax2.imshow(i3.squeeze().cpu().data.numpy()[bstart + j]
                           - output.squeeze().cpu().data.numpy()[j], vmin = -5, vmax = 5, cmap="jet")
                ax3.imshow(i3.squeeze().cpu().data.numpy()[bstart + j],vmin = -1000, vmax=200, cmap="Greys_r")
                plt.ion()
                plt.draw()
                plt.pause(0.01)

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
    return losslist

def evalNet(net, targets, plot=True):
    assert isinstance(targets, dict), "Target should be parsed as dictionaries!"
    assert isinstance(net, network.Net), "Input net is incorrect!"
    assert targets.has_key('032') and targets.has_key('064'), \
            "Dictionary must contain data files with key '032' and '064'"
    
    if (plot):
        fig = plt.figure(1, figsize=[13,6])
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

    net.eval()

    offset = 5
    i2 = targets['032']
    i3 = targets['064']
    last = i2.shape[0] % offset
    indexstart = np.arange(0, i2.shape[0], offset)[0:-1]
    indexstop = indexstart + offset
    i2 = Variable(torch.from_numpy(i2))
    i3 = Variable(torch.from_numpy(i3))
    output = None

    for i in xrange(len(indexstart)):
        bstart = indexstart[i]
        bstop = indexstop[i]

        sl = net.forward(i2[bstart:bstop].cuda(), i3[bstart:bstop].cuda())
        if output is None:
            output = sl.data.cpu().numpy()
        else:
            output = np.concatenate((output, sl.data.cpu().numpy()), 0)

    if last != 0:
        if last == 1:
            bstart = indexstop[-1] - 1
            bstop = bstart + 2
        else:
            bstart = indexstop[-1]
            bstop = bstart + last

        sl = net.forward(i2[bstart:bstop].cuda(), i3[bstart:bstop].cuda())
        np.concatenate((output, sl.data.cpu().numpy()[-1].reshape(
            [1, sl.data.size(1), sl.data.size(2)])), 0)

    if (plot):
        for i in xrange(output.shape[0]):
            plt.ion()
            ax1.imshow(output[i], cmap='Greys_r')
            ax2.imshow(targets['064'][i], cmap='Greys_r')
            plt.draw()
            plt.pause(0.2)

    return output

def main(parserargs):
    logging.getLogger(__name__).log(20, "Start running batch with options: %s"%parserargs)

    #=========================
    # Error check
    #---------------------
    assert a.epoch >= 0, "Epoch must be positive or zero!"
    if (a.usecuda):
        assert torch.has_cudnn, "CUDA is not supported on this machine!"
    if (a.checkpoint):
        if (a.checkpoint is None):
            a.checkpoint = "checkpoint_E%03d"%(a.epoch + 1)
        assert os.path.isfile(a.checkpoint), "Check point file not found!"
        print "Loading checkpoint..."
        logging.getLogger(__name__).log(20, "Loading checkpoint...")
        net = torch.load(a.checkpoint)
    else:
        net = network.Net()

    networkpath = "network_E%03d"%(a.epoch)
    if (os.path.isfile(networkpath)):
        print "Find existing network, loading from %s..."%(networkpath)
        logging.getLogger(__name__).log(20, "Find existing network, loading from %s..."%(networkpath))
        net = torch.load(networkpath)

    #=========================
    # Train
    #---------------------
    if (a.train):
        logging.getLogger(__name__).log(20, "Start training network with %d substeps..."%a.steps)
        assert os.path.isdir(a.input[0]), "Input directory does not exist!"
        b = BatchLoader(a.input[0])
        if a.usecuda:
            logging.getLogger(__name__).log(20, "Using CUDA")
            net.cuda()
        net.zero_grad()

        l = train(net, b, trainsteps=a.steps, epoch=a.epoch, plot=a.plot)
        if a.plot:
            plt.plot(l)
            plt.show()
        else:
            print "Saving figure..."
            logging.getLogger(__name__).log(10, "Saving figure...")
            mpl.use('Agg')
            fig = plt.figure()
            fig.set_tight_layout(True)
            ax1 = fig.add_subplot(111)
            ax1.set_title("Training network Epoch %03d"%(a.epoch + 1))
            ax1.set_xlabel("Step")
            ax1.set_ylabel("Loss")
            ax1.plot(l)
            fig.savefig("fig_E%03d.png"%(a.epoch))

    #==================================
    # Evaluation
    #---------------------------
    # netfile = "network_E003"
    # if (os.path.isfile(netfile)):
    #     net = torch.load(netfile)
    #     evalNet(net, b, True)3cfn
    else:
        assert len(a.input) == 2, "Input should follow format [input032] [input064]"
        assert os.path.isfile(a.input[0]) and os.path.isfile(a.input[1]), \
                "Inputs does not exist!"
        assert (a.output), "Output must be specified!"
        logging.getLogger(__name__).log(10, "Start evaluation...")

        im32 = np.load(a.input[0])
        im64 = np.load(a.input[1])
        targets = {'032': im32, '064':im64}

        output = evalNet(net, targets, a.plot)
        if (a.output.find('.npy') > 0):
            np.save(a.output, output)
        elif (a.output.find('.nii')) > 0:
            from algorithm import NpToNii
            NpToNii(output, a.output)

    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training reconstruction from less projections.")
    parser.add_argument("input", metavar='input', action='store', nargs="+",
                        help="Train/Target input", type=str)
    parser.add_argument("-o", metavar='output', dest='output', action='store', type=str, default=None,
                        help="Set where to store outputs for eval mode")
    parser.add_argument("-p", dest='plot', action='store_true', default=False,
                        help="Select whether to disply the plot for stepwise loss")
    parser.add_argument("-e", "--epoch", dest='epoch', action='store', type=int, default=0,
                        help="Select network epoch.")
    parser.add_argument("-s", "--steps", dest='steps', action='store', type=int, default=1000,
                        help="Specify how many steps to run per epoch.")
    parser.add_argument("--load", dest='checkpoint', action='store', default='',
                        help="Specify network checkpoint.")
    parser.add_argument("--useCUDA", dest='usecuda', action='store_true',default=False,
                        help="Set whether to use CUDA or not.")
    parser.add_argument("--train", dest='train', action='store_true', default=False,
                        help="Set whether to train or evaluate, default is eval")
    parser.add_argument("--train-params", dest='trainparams', action='store', type=dir, default=None,
                        help="Path to a file with dictionary of training parameters written inside")
    parser.add_argument("--log", dest='log', action='store', type=str, default=None,
                        help="If specified, all the messages will be written to the specified file.")
    a = parser.parse_args()

    if (a.log is None):
        if (not os.path.isdir("./Backup/Log")):
            os.mkdir("./Backup/Log")
        a.log = "./Backup/Log/run%03d.log"%(a.epoch)
    logging.basicConfig(format="[%(asctime)-12s - $(levelname)s] %(message)s", filename=a.log)

    main(a)

    # main()
