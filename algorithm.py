
import numpy as np

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
    pass

def ExtractPatchIndexs(im, window, overlap):
    """
    Descriptions
    ------------
      Return a list of index which specify patches. If the specified patch dimension is cannot divide original
      dimension to integer, the last series of patch will have shifted stride.

    :param np.ndarray im:
    :param tuple window:
    :param tuple overlap:
    :return:
    """
    # initialize
    assert type(im) == np.ndarray or type(im) == list
    assert len(window) == len(overlap), "Dimension mismatch!"
    assert all([overlap[i] < window[i] for i in xrange(len(window))]), "Window must be larger than overlap"
    assert type(overlap[0]) == int and type(window[0]) == int, "Window and overlay must be integer!"

    if (type(im) == np.ndarray):
        s = im.shape
    else:
        assert len(im) == len(window), "Dimension mismatch"
        s = im

    # Calculate patch list dimension
    pdim = []
    for i in xrange(len(s)):
        pdim.append((s[i] - window[i])/(window[i] - overlap[i]) + 1
                    + int((s[i] - overlap[i]) % (window[i] - overlap[i]) != 0)
                    # - window[i]/(window[i] - overlap[i]) if s[i] - window[i] > (window[i] - overlap[i]) else 0
                    )
    patches = []
    for i in xrange(len(pdim)):
        arr = np.array([np.arange(0, s[i], (window[i] - overlap[i]))[0:pdim[i]],
                        np.arange(0, s[i], (window[i] - overlap[i]))[0:pdim[i]] + window[i]])
        # Handle last pair
        if (s[i] % (window[i] - overlap[i]) != 0 or s[i] % window[i] != 0):
            arr[0][-1] = s[i] - 1 -window[i] if (s[i] - 1 - window[i] > 0) else 0
            arr[1][-1] = s[i]
        patches.append(arr.transpose())
    return patches

def test():
    dim = [1, 512, 512]
    windows = [1, 32, 512]
    overlap = [0, 8, 8]
    print ExtractPatchIndexs(dim, window=windows, overlap=overlap)

if __name__ == '__main__':
    test()

