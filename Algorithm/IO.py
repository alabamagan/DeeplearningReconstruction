import numpy as np
import fnmatch
import os, sys, gc
import SimpleITK as sitk

def SliceImage(im, dim):
    """

    :param im:
    :param dim:
    :return:
    """
    assert im % 2 == 1, "dim must be odd"

    out = []
    for i in xrange(dim/2, im.shape[0] - dim/2):
        out.append(im[i - dim/2, i + dim/2])

    return out

def SliceAlternate(im1, im2):
    """
    Discriptions
    ------------
      This function returns a list of images having a dimension 2*x*y with
      x, y being the non-axial dimensions of the inputimages. If input image
      has different dimensions, zero padding will be used to make it equal.

      Assuming input has the dimension [z, x, y]

    :param im1:1t
    :param im2:
    :return:
    """

    if (im1.shape != im2.shape):
        xPad = im1.shape[1] - im2.shape[1]
        yPad = im1.shape[2] - im2.shape[2]
        if xPad < 0:
            im1 = np.pad(im1, [(0, 0),
                               (abs(xPad)/2, abs(xPad) - abs(xPad)/2),
                               (0, 0)])
        else:
            im2 = np.pad(im2, [(0, 0),
                               (abs(xPad) / 2, abs(xPad) - abs(xPad) / 2),
                               (0, 0)])
        if yPad < 0:
            im1 = np.pad(im1, [(0, 0),
                               (0, 0),
                               (abs(yPad) / 2, abs(yPad) - abs(yPad) / 2)])
        else:
            im2 = np.pad(im2, [(0, 0),
                               (0, 0),
                               (abs(yPad) / 2, abs(yPad) - abs(yPad) / 2)])

    z, x, y = im1.shape
    out = []
    temp = np.zeros([2 * z, x, y])
    temp[::2,:,:]  = im1
    temp[1::2,:,:] = im2
    for i in xrange(z):
        out.append(temp[i:i+2, :,:])
    return out

def NiiToNpy(infilename, outfilename):
    """
    Descriptions
    ------------
      Convert an Nii image to a npy file.

    :param str infilename:  Path to the input nii file
    :param str outfilename: Desired output path
    :return:
    """
    import SimpleITK as sitk

    if infilename.split('.')[-1] != "nii" and infilename.split('.')[-1] != "gz":
        raise AssertionError("Input file is not Nifty!")


    im = sitk.GetArrayFromImage(sitk.ReadImage(infilename))
    np.save(outfilename, im)
    print "Saving ", outfilename

    del im
    gc.collect()
    pass

def NpToNii(array, outfilename):
    """

    :param np.ndarray array:
    :param str outfilename:
    :return:
    """

    assert isinstance(array, np.ndarray), "First argument must be np.ndarray!"
    assert isinstance(outfilename, str), "Output filename must be a string"

    import SimpleITK as sitk
    if not( outfilename.find(".nii") > 0):
        outfilename += ".nii.gz"

    im = sitk.GetImageFromArray(array)
    sitk.WriteImage(im, outfilename)

    print "Saving ", outfilename
    del im, array
    gc.collect()
    pass

def ConvertAllNpyToNii(directory, output = 'output', selectprojection=None):
    """
    Description
    -----------
      Convert all numpy saved slice into nii.gz. The numpy data should have the naming
      format [prefix]_[projection]_S[03d slice number].npy. This method uses the
      batch loader from pytorch.

    :param directory:
    :param output:
    :return:
    """
    import multiprocessing as mp
    import os

    assert os.path.isdir(directory), "No such directory"
    if not os.path.isdir(output):
        os.mkdir(output)

    ps = []
    pool = mp.Pool(processes=4)
    from dataloader import BatchLoader
    b = BatchLoader(os.path.abspath(directory))

    for i in xrange(len(b)):
        if selectprojection is None:
            for key in b.recon_projection_numbers:
                dst = os.path.abspath(output) + "/" + b.unique_sample_prefix[i] + "_%s.nii.gz"%key
                p = pool.apply_async(NpToNii, args=[b[i][key], dst])
                ps.append(p)
        elif isinstance(selectprojection, list):
            for key in selectprojection:
                dst = os.path.abspath(output) + "/" + b.unique_sample_prefix[i] + "_%s.nii.gz"%key
                p = pool.apply_async(NpToNii, args=[b[i][key], dst])
                ps.append(p)
        else:
            dst = os.path.abspath(output) + "/" + b.unique_sample_prefix[i] + "_%s.nii.gz"%selectprojection
            p = pool.apply_async(NpToNii, args=[b[i][selectprojection], dst])
            ps.append(p)

    for p in ps:
        p.wait()
        del p


class NiiDataLoader(object):
    def __init__(self, rootdir, tonumpy = False):
        super(NiiDataLoader, self).__init__()
        self._rootdir = os.path.abspath(rootdir)
        self._ParseRootDir()
        self._tonumpy = tonumpy
        self._cache = {}

    def __getitem__(self, item):
        """

        :param item:
        :return:
        """
        assert item < len(self.unique_prefix), "Exceed length!"

        if self._cache.has_key(item):
            return self._cache[item]

        d = {}
        for keys in self.types:
            p = self._rootdir + "/" + self.unique_prefix[item] + "_%s.nii.gz"%keys
            if not(os.path.isfile(p)):
                continue

            im = sitk.ReadImage(p)
            if (self._tonumpy):
                temp = im
                im = sitk.GetArrayFromImage(temp)
                del temp
            d[keys] = im
        self.Cache(item, d)
        return d

    def __len__(self):
        return len(self.unique_prefix)


    def Cache(self, item, obj):
        if (sys.getsizeof(self._cache) > 1e6):
            temp =  self._cache.pop()
            del temp
        self._cache[item] = obj

    def PrintLoadable(self):
        for i in xrange(self.__len__()):
            pref = self.unique_prefix[i]
            suff = []
            for keys in self.types:
                if (os.path.isfile(self._rootdir + "/" + pref + "_%s.nii.gz"%keys)):
                    suff.append(keys)
            print pref, ": ", suff
        print "Length: ", self.__len__()

    def _ParseRootDir(self):
        assert os.path.isdir(self._rootdir)

        files = os.listdir(self._rootdir)
        files = fnmatch.filter(files, "*.nii.gz")

        prefix = [ff.split('_')[0] for ff in files]
        types = [ff.split('_')[1].replace('.nii.gz', '') for ff in files]
        self.unique_prefix = list(set(prefix))
        self.types = list(set(types))

        self.unique_prefix.sort()
        self.types.sort()