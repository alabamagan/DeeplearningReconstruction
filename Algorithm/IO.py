import numpy as np
import gc

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

def ConvertAllNpyToNii(directory, output = 'output'):
    import multiprocessing as mp
    import os

    assert os.path.isdir(directory), "No such directory"
    if not os.path.isdir(directory + "/output"):
        os.makedirs(directory + "/output")

    ps = []
    pool = mp.Pool(processes=8)
    fs = os.listdir(directory)
    for fn in fs:
        if (fn.find('.npy') != -1):
            tar = directory + "/" + fn
            dst = directory + "/" + output + "/" + fn.replace('.npy', '.nii.gz')
            print "Working on ", tar
            p = pool.apply_async(NpToNii, args=[np.load(tar), dst])
            ps.append(p)

    for p in ps:
        p.wait()
        del p