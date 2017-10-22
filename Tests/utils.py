import os
import visdom
import fnmatch
import numpy as np
import sys
from dataloader import BatchLoader

vis = visdom.Visdom(port=80)

def main(*args):
    dir = args[0][1]

    files = os.listdir(dir)
    files = fnmatch.filter(files, "*ori*npy")
    files.sort()

    for j in xrange(files.__len__()):
        s = np.load(dir + "/" + files[j])
        s = np.array(s, dtype=float)
        s -= s.min()
        s /= s.max()
        vis.text(files[j], win="filename")
        vis.image(s, win="Fuck")

    pass

if __name__ == '__main__':
    main(sys.argv)