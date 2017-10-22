import os
import fnmatch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader

class BatchLoader(Dataset):
    def __init__(self, rootdir):
        self.root_dir = rootdir
        self._ParseRootdir()

    def __getitem__(self, index):
        if type(index) == tuple:
            idx, idy = index
        else:
            idx = index
            idy = None

        out = {}
        sample = self.unique_sample_prefix[idx]
        ff = os.listdir(self.root_dir)

        for suffix in self.recon_projection_numbers:
            fs = fnmatch.filter(ff, sample + "_" + suffix + "*")
            fs.sort()

            slicenum = len(fs)
            filename = [self.root_dir + "/" + sample + "_" + suffix + "_S%03d.npy"%i for i in xrange(slicenum)]
            if len(filename) == 0:
                continue

            if (idy is None):
                # Return whole image
                images = [np.load(f) for f in filename]
                images = [im.reshape(1, im.shape[0], im.shape[1]) for im in images]
                out[suffix] = np.concatenate(images, 0)
            else:
                out[suffix] = np.load(filename[idy])

        return out
        
    def __call__(self, num):
        """
        Description
        -----------
          This function will return the specified amount of samples randomly drawn from the 
          rootdir.
          
        :param: int num Number of random drawn samples
        :return:
        """

        assert isinstance(num, int), "Call with index"
        assert num > 0, "Why would you want zero samples?"

        out = {}

        # Randomly select indexes
        l = np.random.randint(0, len(self.unique_sample_prefix), num)
        for index in l:
            ll = None
            for suffix in self.recon_projection_numbers:
                fn = self.unique_sample_prefix[index] + "_" + suffix

                fs = os.listdir(self.root_dir)
                fs = fnmatch.filter(fs, fn + "*")
                numOfSlice = len(fs)

                if (ll is None):
                    ll = np.random.randint(0, numOfSlice)

                ff = self.root_dir + "/" + fn + "_S%03d.npy"%ll
                im = np.load(ff)
                im = im.reshape(1, im.shape[0], im.shape[1])

                if not out.has_key(suffix):
                    out[suffix] = im
                else:
                    out[suffix] = np.concatenate([im, out[suffix]], 0)

        return out




    def _ParseRootdir(self):
        """
        Description
        -----------
          Process the root dir and identify unique sample

        :return:
        """

        filenames = fnmatch.filter(os.listdir(self.root_dir), "*.npy")
        recon_projection_numbers = [name.split('_')[1].split('.')[0] for name in filenames]
        filenames = [name.split('_')[0] for name in filenames]
        filenames = list(set(filenames))
        assert len(filenames) > 0, "No npy files were found in root directory: %s"%self.root_dir

        self.unique_sample_prefix = filenames
        self.recon_projection_numbers = set(recon_projection_numbers)

        self.length = len(self.unique_sample_prefix)
        pass


    def __len__(self):
        return self.length
