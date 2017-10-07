import os
import fnmatch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader

class BatchLoader(Dataset):
    def __init__(self, rootdir):
        self.root_dir = rootdir
        self._ParseRootdir()
        self._kernelSize = [32, 32]
        self._compare = ['064', '128']

    def __getitem__(self, index):
        """
        Description
        -----------
          Do NOT use this method
        :param index:
        :return:
        """
        raise AttributeError("Do NOT use this method")

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
        assert num >= self.length, "That few samples are not good for you son."

        #========================================
        # Randomly select some patches
        #--------------------------------
        patches = []
        res = int(num % self.length)
        drawPatches = int(num/self.length)
        ff = os.listdir(self.root_dir)
        for index in xrange(self.length):
            # Get number of slice
            prefix = self.unique_sample_prefix[index]
            fs = fnmatch.filter(ff, prefix + "_" + self._compare[0] + "*")
            fs.sort()
            numOfSlice = len(fs)

            # Get slice bounds
            im0 = np.load(self.root_dir + "/" + fs[0])
            bound = im0.shape

            # Define patches, assume dimension of the same image are always the same
            sliceToPatchesStart = np.array(
                [np.random.randint(0, bound[0] - self._kernelSize[0], size=drawPatches),
                 np.random.randint(0, bound[1] - self._kernelSize[1], size=drawPatches)]
            ).T
            sliceToPatchesStop = np.copy(sliceToPatchesStart)
            sliceToPatchesStop[:,0] += self._kernelSize[0]
            sliceToPatchesStop[:,1] += self._kernelSize[1]
            slicePathces = np.concatenate([sliceToPatchesStart, sliceToPatchesStop], -1) # [xmin, ymin, xmax, ymax]

            # Define which slice the patch is drawn from
            patchDict = {}
            for patchBounds in slicePathces:
                try:
                    s = np.random.randint(0, numOfSlice)
                    patchDict[s].append(patchBounds)
                except KeyError:
                    patchDict[s] = []
                    patchDict[s].append(patchBounds)

            # Actually drawn from slices
            for key in patchDict:
                im0 = np.load(self.root_dir + "/" + prefix + "_" + self._compare[0] + "_S%03d.npy"%int(key))
                im1 = np.load(self.root_dir + "/" + prefix + "_" + self._compare[1] + "_S%03d.npy"%int(key))
                diff = im1 - im0
                extractedPatches = []
                for bounds in patchDict[key]:
                    patch = {}
                    patch[self._compare[0]] = im0[bounds[0]:bounds[2] + 1, bounds[1]:bounds[3] + 1]
                    patch[self._compare[1]] = im1[bounds[0]:bounds[2] + 1, bounds[1]:bounds[3] + 1]
                    patch['diff'] = diff[bounds[0]:bounds[2] + 1, bounds[1]:bounds[3] + 1]
                    extractedPatches.append(patch)
                patches.extend(extractedPatches)

        #===============================================================
        # Remaining patches will be draw from first slice of images
        #-----------------------------------------------------------
        if (res != 0):
            sliceToPatchesStart = np.array(
                [np.random.randint(0, bound[0] - self._kernelSize[0], size=res),
                 np.random.randint(0, bound[1] - self._kernelSize[1], size=res)]
            ).T
            sliceToPatchesStop = np.copy(sliceToPatchesStart)
            sliceToPatchesStop[:,0] += self._kernelSize[0]
            sliceToPatchesStop[:,1] += self._kernelSize[1]
            slicePathces = np.concatenate([sliceToPatchesStart, sliceToPatchesStop], -1) # [xmin, ymin, xmax, ymax]

            patchDict = {}
            for patchBounds in slicePathces:
                try:
                    s = np.random.randint(0, self.__len__())
                    patchDict[s].append(patchBounds)
                except KeyError:
                    patchDict[s] = []
                    patchDict[s].append(patchBounds)


            for key in patchDict:
                prefix = self.unique_sample_prefix[key]
                im0 = np.load(self.root_dir + "/" + prefix + "_" + self._compare[0] + "_S%03d.npy"%int(0))
                im1 = np.load(self.root_dir + "/" + prefix + "_" + self._compare[1] + "_S%03d.npy"%int(0))
                diff = im1 - im0
                extractedPatches = []
                for bounds in patchDict[key]:
                    patch = {}
                    patch[self._compare[0]] = im0[bounds[0]:bounds[2] + 1, bounds[1]:bounds[3] + 1]
                    patch[self._compare[1]] = im1[bounds[0]:bounds[2] + 1, bounds[1]:bounds[3] + 1]
                    patch['diff'] = diff[bounds[0]:bounds[2] + 1, bounds[1]:bounds[3] + 1]
                    extractedPatches.append(patch)
                patches.extend(extractedPatches)


        return patches

    def _SetKernelSize(self, size):
        self._kernelSize = size


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

    def SetTrainMode(self, train):
        assert isinstance(train, bool), "Argument must be bool"
        self.train = train
        self._ParseRootdir()

    def __len__(self):
        return self.length
