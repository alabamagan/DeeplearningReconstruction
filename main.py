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
import visdom
from dataloader import BatchLoader

#============================================
# Prepare global logger
logging.getLogger(__name__).setLevel(10)
vis = visdom.Visdom(server='http://223.255.146.2')

#============================================
# Target key
global targetkey
targetkey=None

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
    global targetkey

    net.train()
    net.current_epoch = epoch

    optimizer = None

    criterion = torch.nn.MSELoss().cuda()
    criterion.size_average = True
    normalize = torch.nn.MSELoss().cuda()
    normalize.size_average = True


    #============================================
    # Actual train phase
    #-------------------------------------
    for i in xrange(trainsteps):
        # index = np.random.randint(0, len(b))
        sample = b(5)
        i3 = sample[targetkey]
        gt = sample['ori']

        if a.invertmask:
            mk = np.logical_not(sample['msk']) # inverted mask
        else:
            mk = sample['msk']
        mk = np.array(mk, dtype=np.uint8)
        gt = Variable(torch.from_numpy(gt)).float()
        i3 = Variable(torch.from_numpy(i3)).float()
        mk = torch.from_numpy(mk)

        if (a.usecuda):
            gt = gt.cuda()
            i3 = i3.cuda()
            mk = mk.cuda()

        output = net.forward(i3.cuda())
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
                                         {'params': net.miscParams.parameters(), 'lr':1e-1}
                                         ])
            optimizer.zero_grad()
        else:
           # Decay learning rate
           if (a.decay != 0):
            for pg in optimizer.param_groups:
               pg['lr'] = pg['lr'] * np.exp(-i * float(a.epoch)  * a.decay / float(trainsteps))


        loss = criterion((output.squeeze())[mk], (gt)[mk]) / normalize(i3[mk], gt[mk])
        print "[Step %04d] Loss: %.010f"%(i, loss.data[0])
        logging.getLogger(__name__).log(20, "[Step %04d] Loss: %.010f"%(i, loss.data[0]))
        net.current_step += 1
        net.loss_list.append(loss.data[0])
        loss.backward()


        optimizer.step()

        #======================================
        # Plot for visualization of result
        #----------------------------------
        if (plot):
            paramstext = "<h2>Step %03d </h2> <br>"%i + \
                         "<h3>Col Means: " + \
                         ", ".join([str(p.data[0]) for p in net.linearModules.parameters()]) + \
                         "</h3>"

            # Setting display value range
            displayrangeIm = [-1000, 400]
            displayrangeDiff = [-15, 15]

            # Normalization for display
            normIm = lambda inIm: inIm.clip(displayrangeIm[0], displayrangeIm[1])/\
                                  float(displayrangeIm[1] - displayrangeIm[0])
            normDiff = lambda inIm: inIm.clip(displayrangeDiff[0], displayrangeDiff[1])/\
                                  float(displayrangeDiff[1] - displayrangeDiff[0])


            ims = []
            ims.append((output.squeeze().unsqueeze(1).data.cpu() -
                        i3.squeeze().unsqueeze(1).data.cpu()).numpy()) # Processed True Diff
            ims.append((gt.squeeze().unsqueeze(1).data.cpu() -
                        i3.squeeze().unsqueeze(1).data.cpu()).numpy()) # Ground True Diff
            ims.append(ims[1] - ims[0])                                # Difference
            ims.append((i3.squeeze().unsqueeze(1).data.cpu()).numpy()) # Original Image
            ims.append((output.squeeze().unsqueeze(1).data.cpu()).numpy()) # Processed Image

            ims = [normDiff(im) for im in ims[0:3]] + [normIm(im) for im in ims[3:5]]
            ims = [im - im.min() for im in ims]
            ims = [im / im.max() for im in ims]


            vis.text(paramstext, env="Results", win="ParamsWindow")
            [vis.images(ims[k +3], nrow=1, env="Results", win="ImWindow%i"%k) for k in xrange(len(ims) - 3)]

            losslist = np.array(net.loss_list)
            vis.line(losslist, np.arange(len(net.loss_list)), env="Plots", win="TrainingLoss")



        if (i % 100 == 0):
            torch.save(net, "checkpoint_E%03d"%(epoch + 1))

        # Free some meory
        del sample, i3, gt
        gc.collect()

    print "======================= End train epoch %03d ======================="%(epoch + 1)
    print "average loss: ", losslist.mean()
    print "final loss: ", losslist[-1]
    logging.getLogger(__name__).log(20,"======================= End train epoch %03d ======================="%(epoch + 1))
    logging.getLogger(__name__).log(20,"Average loss: %.05f"%losslist.mean())
    torch.save(net, "network_E%03d"%(epoch + 1))
    return net.loss_list

def evalNet(net, targets, plot=True):
    global targetkey

    #========================================
    # Error check
    #---------------------------------------
    assert isinstance(targets, dict), "Target should be parsed as dictionaries!"
    assert isinstance(net, network.Net), "Input net is incorrect!"
    assert targets.has_key(targetkey) and targets.has_key('ori'), \
            "Dictionary must contain data files with key %s and ori"%targetkey

    # Set network to evaluation mode
    net.eval()

    # Set the batch number the network can take without overflowing GPU memory
    offset = 5
    oi3 = targets[targetkey]
    mk = targets['msk']

    # Calculate the interval indexes
    last = oi3.shape[0] % offset
    if last == 0:
        indexstart = np.arange(0, oi3.shape[0], offset)
    else:
        indexstart = np.arange(0, oi3.shape[0], offset)[0:-1]
    indexstop = indexstart + offset

    output = None
    for i in xrange(len(indexstart)):
        bstart = indexstart[i]
        bstop = indexstop[i]

        i3 = Variable(torch.from_numpy(oi3[bstart:bstop]), requires_grad=False)
        if (a.usecuda):
            i3 = i3.float().cuda()

        sl = net.forward(i3)
        if output is None:
            output = sl.data.cpu().numpy()
        else:
            output = np.concatenate((output, sl.data.cpu().numpy()), 0)

        # Free some GRAM
        del sl, i3

    if last != 0:
        if last == 1:
            bstart = indexstop[-1] - 1
            bstop = bstart + 2

            i3 = Variable(torch.from_numpy(oi3[bstart:bstop]), requires_grad=False)

            if (a.usecuda):
                i3 = i3.float().cuda()

            sl = net.forward(i3)
            output =np.concatenate((output, sl.data.cpu().numpy()[-1].reshape(
                                [1, sl.data.size(1), sl.data.size(2)])), 0)

        else:
            bstart = indexstop[-1]
            bstop = bstart + last

            i3 = Variable(torch.from_numpy(oi3[bstart:bstop]), requires_grad=False)

            if (a.usecuda):
                i3 = i3.float().cuda()

            sl = net.forward(i3)
            output =np.concatenate((output, sl.data.cpu().numpy()), 0)

        # Free some GRAM
        del sl, i3

    # Calculate loss with np if ori exist
    loss = None
    if (targets.has_key('ori')):
        loss =  np.sum(np.abs(targets['ori'] - output)[mk > 0]**2) / \
                np.sum(np.abs(targets['ori'] - targets[targetkey])[mk > 0]**2)
        LogPrint("Calculated loss: %.05f"%loss, 20)



    return output, loss

def main(parserargs):
    global targetkey
    targetkey = parserargs.targetkey

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

    if a.usecuda:
        logging.getLogger(__name__).log(20, "Using CUDA")
        net.cuda()

    #=========================
    # Train
    #---------------------
    if (a.train):
        logging.getLogger(__name__).log(20, "Start training network with %d substeps..."%a.steps)
        assert os.path.isdir(a.input[0]), "Input directory does not exist!"
        b = BatchLoader(a.input[0])
        net.zero_grad()

        # Parse params
        if (a.trainparams != None):
            import ast
            trainparams = ast.literal_eval(a.trainparams)
        else:
            trainparams = None

        l = train(net, b, trainsteps=a.steps, epoch=a.epoch, plot=a.plot, params=trainparams)


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

            b = BatchLoader(a.input[0])
            assert len(b) != 0, "Nothing in directory!"

            # Default output path
            if (a.output is None):
                outdir = "%.05f"%np.random.rand()
            else:
                outdir = a.output

            if not (os.path.isdir(outdir)):
                outdir = a.input[0] + "/Generated/"

            if not os.path.isdir(outdir):
                os.makedirs(outdir)

            ostream = file(outdir + "/results.txt", 'w')
            losslist = []
            for i in xrange(len(b)):
                images = b[i]
                name = b.unique_sample_prefix[i]

                output, loss = evalNet(net, images, a.plot)
                ostream.write(name + " " + str(loss) + "\r\n")

                # Save the output
                from Algorithm.IO import NpToNii
                NpToNii(output, outdir + "/" + b.unique_sample_prefix[i] + "_processed.nii.gz")
                NpToNii(images[targetkey], outdir + "/" + b.unique_sample_prefix[i] + "_%s.nii.gz"%targetkey)
                logging.getLogger(__name__).log(10, "Saving to " + outdir + "/" + b.unique_sample_prefix[i] + "_processed.nii.gz")
                losslist.append(loss)

            logging.getLogger(__name__).log(20, "=============== Eval E%03d End==============="%(a.epoch + 1))
            logging.getLogger(__name__).log(20, "Average loss: %.05f"%(np.mean(losslist)))
            ostream.write("Average loss: %.05f"%(np.mean(losslist)))
            ostream.close()


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
                from Algorithm.Utils import NpToNii
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
    parser.add_argument("--targetkey", dest='targetkey', action='store', default='128', type=str,
                        help="The projection unique identification key. Default to be '128'")
    parser.add_argument("--invert-mask", dest='invertmask', action='store_true', default=False,
                        help="Invert the mask for loss function. "
                             "Useful for learning background noise suppression. Note that this is best used with"
                             "alternative loss function.")
    parser.add_argument("--vacinity-loss", dest='vacloss', action='store_true', default=False,
                        help="Use alternative loss function.")

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

    # try:
    #     main(a)
    # except:
    #     vis.text("Interupted!", win='ParamsWindow', env="Results")

    # main()
