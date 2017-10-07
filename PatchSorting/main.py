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


        #=================================================
        # Add modules params to optimizer
        #-----------------------------------------------
        if (optimizer == None):

            if params != None:


            optimizer = torch.optim.SGD([{'params': net.convsModules.parameters(),
                                          'lr': convLR, 'momentum':1e-2, 'dampening': 1e-2}
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
        if (not os.path.isdir("../Backup/Log")):
            os.mkdir("../Backup/Log")
        if (a.train):
            a.log = "../Backup/Log/PatchSort_run%03d.log"%(a.epoch)
        else:
            a.log = "../Backup/Log/PatchSort_eval_%03d.log"%(a.epoch)

    logging.basicConfig(format="[%(asctime)-12s - %(levelname)s] %(message)s", filename=a.log)

    main(a)

    # main()
