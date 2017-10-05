#!/home/lwong/Toolkits/Anaconda2/bin/python
#  import SimpleITK as sitk
import matplotlib as mpl
import network
import torch
from torch.autograd import Variable
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import gc
import logging
import argparse
from dataloader import BatchLoader

#============================================
# Prepare global logger
logging.getLogger(__name__).setLevel(10)

def LogPrint(msg, level=20):
    logging.getLogger(__name__).log(level, msg)
    print msg

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


    #============================================
    # Actual train phase
    #-------------------------------------
    for i in xrange(trainsteps):
        # index = np.random.randint(0, len(b))
        sample = b(30)
        i2 = sample['064']
        i3 = sample['128']
        gt = sample['ori']
        gt = Variable(torch.from_numpy(gt)).float().cuda()
        i2 = Variable(torch.from_numpy(i2)).float()
        i3 = Variable(torch.from_numpy(i3)).float()

        # offset = 10
        # bstart = np.random.randint(0, i2.data.size()[0] - offset)
        # bstop = bstart + offset

        output = net.forward(i2.cuda(), i3.cuda())
        #=================================================
        # Add modules params to optimizer
        #-----------------------------------------------
        if (optimizer == None):
            # Default params
            convLR = 10
            fcLR = 2
            bnLR = 2
            psLR = 2
            poolingLR = 2
            linearLR = 1
            if params != None:
                if params.has_key('convLR'):
                    convLR = params['convLR']
                if params.has_key('fcLR'):
                    fcLR = params['fcLR']
                if params.has_key('bnLR'):
                    bnLR = params['bnLR']
                if params.has_key('psLR'):
                    psLR =params['psLR']
                if params.has_key('poolingLR'):
                    poolingLR = params['poolingLR']
                if params.has_key('linearLR'):
                    linearLR = params['linearLR']

            optimizer = torch.optim.SGD([{'params': net.convsModules.parameters(),
                                          'lr': convLR, 'momentum':1e-2, 'dampening': 1e-2},
                                         {'params': net.psModules.parameters(),
                                          'lr': psLR, 'momentum':1e-3, 'dampling':1e-2},
                                         {'params': net.poolingLayers.parameters(),
                                          'lr': poolingLR, 'momentum':1e-3, 'dampling':1e-2},
                                         {'params': net.fcModules.parameters(), 'lr': fcLR},
                                         {'params': net.bnModules.parameters(), 'lr': bnLR},
                                         {'params': net.linearModules.parameters(), 'lr': linearLR},
                                         {'params': net.linear1.parameters(),
                                          'lr': linearLR, 'momentum':0, 'dampening':1e-5}
                                         ])
            optimizer.zero_grad()
        else:
           # Decay learning rate
           if (a.decay != 0):
            for pg in optimizer.param_groups:
               pg['lr'] = pg['lr'] * np.exp(-i * float(a.epoch)  * a.decay / float(trainsteps))

        #============================================
        # Pre-train phase
        #-------------------------------------
        if (i == 0 and epoch == 0 and a.pretrain):
            if (os.path.isfile("pretrain_checkpoint_E%03d"%(epoch + 1))):
                net.load_state_dict(torch.load("pretrain_checkpoint_E%03d"%(epoch + 1)))
                LogPrint("Loading pretrain dict")
            else:
                LogPrint(">>>>>>>>>>>>>>> Pre-train Phase <<<<<<<<<<<<<<<<<")
                for j in xrange(500):
                    loss = criterion((output.squeeze()), (gt)) / normalize(i3.float().cuda(), gt)
                    loss.backward()
                    optimizer.step()
                    output = net.forward(i2.cuda(), i3.cuda())
                    LogPrint("[Pretrain %04d] Loss: %.010f"%(j, loss.data[0]))
                LogPrint(">>>>>>>>>>>>>>> Pre-train Phase End <<<<<<<<<<<<<<<<<")
                torch.save(net.state_dict(), "pretrain_checkpoint_E%03d"%(epoch + 1))

        loss = criterion((output.squeeze()), (gt)) / normalize(i3.float().cuda(), gt)
        print "[Step %04d] Loss: %.010f"%(i, loss.data[0])
        logging.getLogger(__name__).log(20, "[Step %04d] Loss: %.010f"%(i, loss.data[0]))
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
                # ax1.imshow(gt.squeeze().cpu().data.numpy()[j], vmin =-1000, vmax=-500, cmap="Greys_r")
                ax1.imshow(i3.squeeze().cpu().data.numpy()[j]
                           - i2.squeeze().cpu().data.numpy()[j], vmin = -15, vmax = 15, cmap="Greys_r")
                ax2.imshow(i3.squeeze().cpu().data.numpy()[j]
                           - output.squeeze().cpu().data.numpy()[j], vmin = -15, vmax = 15, cmap="Greys_r")
                ax3.imshow(i3.squeeze().cpu().data.numpy()[j] -
                           gt.squeeze().cpu().data.numpy()[j],vmin = -15, vmax=15, cmap="Greys_r")
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
    logging.getLogger(__name__).log(20,"======================= End train epoch %03d ======================="%(epoch + 1))
    logging.getLogger(__name__).log(20,"Average loss: %.05f"%losslist.mean())
    torch.save(net, "network_E%03d"%(epoch + 1))
    return losslist

def evalNet(net, targets, plot=True):
    assert isinstance(targets, dict), "Target should be parsed as dictionaries!"
    assert isinstance(net, network.Net), "Input net is incorrect!"
    assert targets.has_key('128') and targets.has_key('064'), \
            "Dictionary must contain data files with key '128' and '064'"
    
    if (plot):
        fig = plt.figure(1, figsize=[13,6])
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

    net.eval()

    offset = 40
    i2 = targets['064']
    i3 = targets['128']
    last = i2.shape[0] % offset
    if last == 0:
        indexstart = np.arange(0, i2.shape[0], offset)
    else:
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
            sl = net.forward(i2[bstart:bstop].cuda(), i3[bstart:bstop].cuda())
            output =np.concatenate((output, sl.data.cpu().numpy()[-1].reshape(
                                [1, sl.data.size(1), sl.data.size(2)])), 0)

        else:
            bstart = indexstop[-1]
            bstop = bstart + last
            sl = net.forward(i2[bstart:bstop].cuda(), i3[bstart:bstop].cuda())
            output =np.concatenate((output, sl.data.cpu().numpy()), 0)

    # Calculate loss with np if ori exist
    loss = None
    if (targets.has_key('ori')):
        loss =  np.sum(np.abs(targets['ori'] - output)) / \
                np.sum(np.abs(targets['ori'] - targets['064']))
        logging.getLogger(__name__).log(20, "Calculated loss: %.05f"%loss)

    if (plot):
        for i in xrange(output.shape[0]):
            plt.ion()
            ax1.imshow(output[i], cmap='Greys_r')
            ax2.imshow(targets['064'][i], cmap='Greys_r')
            plt.draw()
            plt.pause(0.2)

    return output, loss

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

        # Parse params
        if (a.trainparams != None):
            import ast
            trainparams = ast.literal_eval(a.trainparams)
        else:
            trainparams = None

        l = train(net, b, trainsteps=a.steps, epoch=a.epoch, plot=a.plot, params=trainparams)
        if a.plot:
            plt.plot(l)
            plt.show()
        else:
            print "Saving figure..."
            logging.getLogger(__name__).log(10, "Saving figure...")
            plt.switch_backend('Agg')
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
        if len(a.input) == 1:
            import fnmatch

            assert os.path.isdir(a.input[0]), "Input directory does not exist!"
            logging.getLogger(__name__).log(10, "Start evaluation on directory %s"%a.input[0])

            b = BatchLoader(a.input)

            # Default output path
            if (a.output is None):
                outdir = "%.05f"%np.random.rand()
            else:
                outdir = a.output

            if not (os.path.isdir(outdir)):
                outdir = a.input[0] + "/Generated/"

            if not os.path.isdir(outdir):
                os.makedirs(outdir)

            losslist = []
            for i in xrange(len(b)):
                batchsize  = 30

                images = b[i]

                targets = {'032': images['032'], '064':images['064'], '128': images['128'], 'ori': images['ori']}

                output, loss = evalNet(net, targets, a.plot)



                from algorithm import NpToNii
                NpToNii(output, outdir + fn + "_processed.nii.gz")
                logging.getLogger(__name__).log(10, "Saving to " + outdir + fn + "_processed.nii.gz")
                losslist.append(loss)

            logging.getLogger(__name__).log(20, "=============== Eval E%03d End==============="%(a.epoch + 1))
            logging.getLogger(__name__).log(20, "Average loss: %.05f"%(np.mean(losslist)))


        elif len(a.input) == 2:
            assert os.path.isfile(a.input[0]) and os.path.isfile(a.input[1]), \
                    "Inputs does not exist!"
            assert (a.output), "Output must be specified!"
            logging.getLogger(__name__).log(10, "Start evaluation on one target...")

            im32 = np.load(a.input[0])
            im64 = np.load(a.input[1])
            targets = {'032': im32, '064':im64}

            output = evalNet(net, targets, a.plot)[0]
            if (a.output.find('.npy') > 0):
                np.save(a.output, output)
            elif (a.output.find('.nii')) > 0:
                from algorithm import NpToNii
                NpToNii(output, a.output)

        else:
            print "Wrong number of arguments!"

    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training reconstruction from less projections.")
    parser.add_argument("input", metavar='input', action='store', nargs="+",
                        help="Train/Target input", type=str)
    parser.add_argument("-o", metavar='output', dest='output', action='store', type=str, default=None,
                        help="Set where to store outputs for eval mode")
    parser.add_argument("-p", dest='plot', action='store_true', default=False,
                        help="Select whether to disply the plot for stepwise loss")
    parser.add_argument("-d", "--decayLR", dest='decay', action='store', type=float, default=0,
                        help="Set decay halflife of the learning rates.")
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
    parser.add_argument("--train-params", dest='trainparams', action='store', type=str, default=None,
                        help="Path to a file with dictionary of training parameters written inside")
    parser.add_argument("--log", dest='log', action='store', type=str, default=None,
                        help="If specified, all the messages will be written to the specified file.")
    parser.add_argument("--pretrain", dest='pretrain', action='store_true', default=False,
                        help="Use a set of randomly drawn sample from the training data to do 200 steps pretrain")
    a = parser.parse_args()

    if (a.log is None):
        if (not os.path.isdir("./Backup/Log")):
            os.mkdir("./Backup/Log")
        if (a.train):
            a.log = "./Backup/Log/run%03d.log"%(a.epoch)
        else:
            a.log = "./Backup/Log/eval_%03d.log"%(a.epoch)

    logging.basicConfig(format="[%(asctime)-12s - %(levelname)s] %(message)s", filename=a.log)

    main(a)

    # main()
