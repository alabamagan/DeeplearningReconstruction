import os
import fnmatch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader

class BatchLoader(Dataset):
    def __init__(self, rootdir):
        self.root_dir = rootdir
        self.train = True
        self._ParseRootdir()

    def __getitem__(self, index):
        if type(index) == tuple:
            idx, idy = index
        else:
            idx = index
            idy = None

        if (not self.train):
            out = {}
            sample = self.unique_sample_prefix[idx]
            for suffix in self.recon_projection_numbers:
                filename = self.root_dir + "/" + sample + "_" + suffix + ".npy"
                im = np.load(filename)

                if (idy is None):
                    out[suffix] = im
                else:
                    out[suffix] = im[idy]
        else:
            # Random slice order for training mode
            assert idy != None, "Second index must be provided in training mode"

            out = {}
            sample = self.unique_sample_prefix[idx]
            # Obtain number of files
            fs = os.listdir(self.root_dir)
            fs = fnmatch.filter(fs, sample + "*")
            s = len(fs) / len(self.recon_projection_numbers)
            # Randomly select indexes

            assert idy < s, "Requested to much slices!"
            l =random.sample(range(s), idy)
            for suffix in self.recon_projection_numbers:
                filenames = [self.root_dir + "/" + sample + "_"
                             + suffix + "_S%03d.npy"%slicenum
                            for slicenum in l]
                im = [np.load(fn) for fn in filenames]
                im = [subim.reshape([1, subim.shape[0], subim.shape[1]]) for subim in im]
                im = np.concatenate(im, 0)

                out[suffix] = im


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

        if (not self.train):
            self.length = len(filenames) / len(recon_projection_numbers)
        else:
            self.length = len(self.unique_sample_prefix)
        pass

    def SetTrainMode(self, train):
        assert isinstance(train, bool), "Argument must be bool"
        self.train = train
        self._ParseRootdir()

    def __len__(self):
        return self.length
