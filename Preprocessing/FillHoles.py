import SimpleITK as sitk
import numpy as np
import os
import fnmatch
import visdom

vis = visdom.Visdom(port=80)

def ParseRootdir(dir):
    """
    Description
    -----------
      Process the root dir and identify unique sample

    :param str dir
    :return:
    """

    filenames = fnmatch.filter(os.listdir(dir), "*.npy")
    recon_projection_numbers = [name.split('_')[1].split('.')[0] for name in filenames]
    filenames = [name.split('_')[0] for name in filenames]
    filenames = list(set(filenames))
    assert len(filenames) > 0, "No npy files were found in root directory: %s"%dir

    return filenames

def FillHole2D(array, mask=True):
    """
    Description
    -----------
      Fill holes in 2D space instead of 3D so holes connected to the image boundary
      will still be filled.

    :param np.ndarray array: 2D array to be filled, assume (H, W)
    :param bool mask:        Use circular mask or not
    :return:
    """
    assert isinstance(array, np.ndarray), "Input has to be array"
    assert array.ndim == 2, "Input has to be 2D"

    m = None
    if mask:
         s = array.shape
         r = float(s[1] / s[0])
         x, y, = np.meshgrid(np.arange(0, s[0]), np.arange(0, s[1]))
         m = (x - s[0]/2)**2. + (y - s[0]/2)**2 / r > (s[0] / 2.)**2 + 1

    array[m] = -3024
    dis = np.array(array, dtype=float)
    dis -= dis.min()
    dis /= dis.max()
    # vis.image(dis, win="Before")

    im = sitk.GetImageFromArray(array)
    im = sitk.BinaryThreshold(im,  lowerThreshold=-400, upperThreshold=5000, insideValue=1)
    im = sitk.BinaryFillhole(im)
    im = sitk.BinaryErode(im, 8)
    im = sitk.BinaryDilate(im, 12)

    im = np.array(sitk.GetArrayFromImage(im), dtype=float)
    im[m] = 0

    # vis.image(im, env="main", win="FillHole")
    return im

def FillHole3D(array, mask=True):
    """
    Description
    -----------
      Assume input array is in shape of [B, H, W]

    :param np.ndarray array: 3D array to be filled
    :param bool mask:        Use circular mask or not
    :return:
    """
    assert isinstance(array, np.ndarray), "Input has to be array"
    assert array.ndim == 3, "Input has to be 3D"

    im = []
    for i in xrange(array.shape[0]):
        l_im = FillHole2D(array[i], mask)
        im.append(l_im)

    im = [img.reshape(1, img.shape[0], img.shape[1]) for img in im]
    return np.concatenate(im, 0)

def SaveImage(array, prefix):
    """
    Description
    -----------
      Save the numpy image assuming dimension is (B, H, W) along the batch (i.e. slice)
      direction. Each slices are saved as separated .npy files with the format
      @prefix + "_msk_S%03d"%slicenum

    :param np.ndarray array: Array to be saved, assume (B, H, W)
    :param str prefix:       Prefix of the saved files
    :return:
    """


    assert isinstance(array, np.ndarray)
    assert isinstance(prefix, str)
    assert array.ndim == 3, "Only support 3D images (B, H, W)"

    for i in xrange(array.shape[0]):
        np.save(prefix + "_msk_S%03d"%i, array[i])

def ProcessDirectory(dir):
    """
    Description
    -----------
      Process all the npy files with fill hole methods.

    :param str dir:
    :return:
    """

    assert os.path.isdir(dir), "Directory doesn't exist"

    fns = ParseRootdir(dir)
    assert len(fns) != 0, "Directory is empty"

    result = {}
    for i in xrange(len(fns)):
        print "Working on ", i
        fs = os.listdir(dir)
        fs = fnmatch.filter(fs, fns[i] + "*_128_*")
        fs.sort()
        im = [np.load(dir + "/" + f) for f in fs]
        im = [img.reshape(1, img.shape[0], img.shape[1]) for img in im]
        im = np.concatenate(im, 0)
        im = FillHole3D(im)
        im = np.array(im, dtype=np.uint8)
        result[i] = im

    for i in xrange(len(fns)):
        SaveImage(result[i], dir + "/" + fns[i])


def ShowMaskOnVisdom(dir):
    """
    Description
    -----------
      Show the result in visdom server (env = "main")

    :param dir:
    :return:
    """
    assert os.path.isdir(dir), "Directory doesn't exist"

    fns = ParseRootdir(dir)
    assert len(fns) != 0, "Directory is empty"

    for i in xrange(len(fns)):
        print "Working on ", i
        fs = os.listdir(dir)
        f1 = fnmatch.filter(fs, fns[i] + "*_msk_*")
        f2 = fnmatch.filter(fs, fns[i] + "*_128_*")
        f1.sort()
        f2.sort()


        im1 = [np.load(dir + "/" + f) for f in f1]
        im1 = [np.array(im, dtype=float) for im in im1]
        im1 = [im - im.min() for im in im1]
        im1 = [im / im.max() for im in im1]
        im2 = [np.load(dir + "/" + f) for f in f2]
        im2 = [np.array(im, float) for im in im2]
        im2 = [im - im.min() for im in im2]
        im2 = [im / im.max() for im in im2]

        for j in xrange(len(im1)):
            vis.image(im1[j], win="Im1")
            vis.image(im2[j], win="Im2")


def main():
    ShowMaskOnVisdom("../SIRT_Parallel_Slices/test")

if __name__ == '__main__':
    main()

