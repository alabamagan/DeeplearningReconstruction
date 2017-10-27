import os
import visdom
import fnmatch
import numpy as np
import sys
from dataloader import BatchLoader
import os
vis = visdom.Visdom(port=80)

__all__ = ["ShowImages", "CreateTrainingSamples", "RemoveSpeicfiedSlices"]

def MoveFiles(prefix, fn, postix):
    print "Moving %s to %s..."%(fn, postix)
    os.system("mv " + prefix + "/" + fn + "* " + prefix + "/" + postix)
    return

def ShowImages(*args):
    """
    Description
    -----------
      Show all .npy files in a directory with visdom

    :param args:
    :return:
    """
    dir = args[0][1]

    disrange = [-1000, 400]

    files = os.listdir(dir)
    files = fnmatch.filter(files, "*_064_*npy")
    files.sort()

    for j in xrange(files.__len__()):
        s = np.load(dir + "/" + files[j])
        s = np.array(s, dtype=float)
        s = np.clip(s, disrange[0], disrange[1])
        s -= s.min()
        s /= s.max()
        vis.text(files[j], win="filename")
        vis.image(s, win="Fuck")

    pass

def CreateTrainingSamples(num, dir):
    """
    Description
    ------------
      Draw samples from a set of directories

    :param int num:
    :param str dir:
    :return:
    """

    import fnmatch
    import random
    import multiprocessing as mpi

    assert isinstance(dir, str), "Directory must be a string!"
    assert isinstance(num, int), "Num must be integer!"
    assert os.path.isdir(dir), "Directory doesn't exist!"

    dirfiles = os.listdir(dir)
    npfiles = fnmatch.filter(dirfiles, "*.npy")

    # Count number of samples in folder
    uniqueSamples = [fs.split('_')[0] for fs in npfiles]
    uniqueSamples = list(set(uniqueSamples))

    assert len(uniqueSamples) > num, "Required unique samples greater than" \
                                     "total number of samples in the directory"

    # Create output directory if not exist
    if not os.path.isdir(dir + "/train"):
        os.mkdir(dir + "/train")
    if not os.path.isdir(dir + "/test"):
        os.mkdir(dir + "/test")

    # Choose from original directory
    trainsamples = random.sample(np.arange(len(uniqueSamples)), num)
    trainsamples = [uniqueSamples[i] for i in trainsamples]

    pool = mpi.Pool(processes=8)
    p = []

    # Move files
    for fs in uniqueSamples:
        if fs in trainsamples:
            process = pool.apply_async(MoveFiles, args=[dir, fs, "train"])
            p.append(process)
            # MoveFiles(fs, "train")
        else:
            process = pool.apply_async(MoveFiles, args=[dir, fs, "test"])
            p.append(process)
            # MoveFiles(fs, "test")

    # Wait till job finish
    for process in p:
        process.wait()

    pass

def RemoveSpeicfiedSlices(dir, spec):
    """
    Description
    -----------
      Clean directory and only keep slices and files specified by param spec.
      Spec should either be a directory or a nested list.

      The file should be arranged in the following fashion using space as separator:
      #1  [unique_sample_prefix_1] [start slice_1] [end slice_1]
      #2  [unique_sample_prefix_2] [start slice_2] [end slice_2]
      #3  ...

      The nested list should be arranged in the following fashion:
      [ [unique_sample_prefix_1, start_slice_1, end_slice_1],
        [unique_sample_prefix_2, start_slice_2, end_slice_2],
        ... ]

    :param str dir:
    :param str/list spec:
    :return:
    """

    import re
    import multiprocessing as mpi

    # Create directory to hold removed slices
    if not os.path.isdir(dir + "/removed"):
        os.mkdir(dir + "/removed")

    # Read file into a list
    if isinstance(spec, str):
        temp = []
        for line in file(spec, 'r').readlines():
            temp.append(line.replace('\n', '').split(' '))
        spec = temp

    files = os.listdir(dir)
    uniquesamples = list(set([fs.split('_')[0] for fs in files]))
    specSamples = [s[0] for s in spec]

    # Check if any samples are not in the files
    if len(files) != len(uniquesamples):
        for us in uniquesamples:
            if not (us in specSamples):
                os.system("mv " + us + "* " + dir + "/removed")

    # Identify which files to be moved
    tobemoved = []
    for i in xrange(len(spec)):
        fs = fnmatch.filter(files, spec[i][0] + "*")
        for ff in fs:
            result = re.match(r'.*S([0-0]+).*', ff)
            if result is None:
                print "Pattern error for file: " + ff
                continue

            sliceNum = int(result.group(1))
            if sliceNum < spec[i][1] or sliceNum >= spec[i][2]:
                tobemoved.append(dir + "/" + ff)

    # Move files
    indexlist = range(len(tobemoved))
    indexlist = indexlist[::10000]
    if indexlist[-1] != len(tobemoved) - 1:
        indexlist.append(len(tobemoved) - 1)

    pool = mpi.Pool(processes=6)
    p = []
    for i in xrange(len(indexlist) - 1):
        arg = " ".join(tobemoved[indexlist[i]:indexlist[i+1]])
        com = "mv " + arg + " " + dir + "/removed"
        p.append(pool.apply_async(os.system, arg=[com]))

    for process in p:
        process.wait()

    pass


def CheckDir(dir):
    """
    Description
    -----------
      Check if directory has the correct format

    :param str dir:
    :return:
    """

    assert os.path.isdir(dir), "Directory doesn't exist!"

    files = os.listdir(dir)
    files = fnmatch.filter(files, "*.npy")
    files.sort()

    uniquesamples = list(set([ff.split('_')[0] for ff in files]))
    suffix = list(set([ff.split('_')[1] for ff in files]))
    suffix.sort()
    print suffix

    for f in uniquesamples:
        slices = []
        for suf in suffix:
            fs = fnmatch.filter(files, f + "_" + suf + "_*")
            slices.append(str(len(fs)))
        print f + ": " + ",".join(slices)







