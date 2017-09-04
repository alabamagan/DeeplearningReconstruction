import os
import fnmatch
import numpy as np
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
        for suffix in self.recon_projection_numbers:
            filename = self.root_dir + "/" + sample + "_" + suffix + ".npy"
            im = np.load(filename)

            if (idy is None):
                out[suffix] = im[np.random.randint(0, im.shape[0] - 1)]
            else:
                out[suffix] = im[idy]

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
        self.unique_sample_prefix = filenames
        self.recon_projection_numbers = set(recon_projection_numbers)
        pass

    def __len__(self):
        return len(self.unique_sample_prefix)