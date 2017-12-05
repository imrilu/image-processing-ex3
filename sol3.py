import numpy as np
import scipy
import scipy.signal
from skimage.color import rgb2gray
from scipy.misc import imread as imread
from scipy import linalg as linalg
import matplotlib.pyplot as plt


def read_image(filename, representation):
    """
    A method for reading an image from a path and loading in as gray or in color
    :param filename: The path for the picture to be loaded
    :param representation: The type of color the image will be load in. 1 for gray,
    2 for color
    :return: The loaded image
    """
    im = imread(filename)
    if representation == 1:
        # converting to gray
        im = rgb2gray(im) / 255
    else:
        if representation == 2:
            im = im.astype(np.float64)
            # setting the image's matrix to be between 0 and 1
            im = im / 255
    return im


def create_gaussian_line(size):
    """
    A helper method for creating a gaussian kernel with the input size
    :param size: The size of the output dimension of the gaussian kernel
    :return: A discrete gaussian kernel
    """
    bin_arr = np.array([1, 1])
    org_arr = np.array([1, 1])
    # TODO: what if size==1 - should return kernel with [1] ?
    if (size == 1):
        # special case, returning a [1] matrix
        return np.array([1])
    for i in range(size-2):
        # iterating to create the initial row of the kernel
        bin_arr = scipy.signal.convolve(bin_arr, org_arr)
    bin_arr = np.divide(bin_arr, bin_arr.sum())
    bin_arr = np.reshape(bin_arr, (1,size))
    return bin_arr


def expand(im, filter_vec=None):
    new_expand = np.zeros(shape=(int(im.shape[0]*2), int(im.shape[1]*2)))
    new_expand[::2,::2] = im
    if filter_vec is None:
        kernel = create_gaussian_line(3)
    else:
        kernel = filter_vec
    new_expand = scipy.signal.convolve2d(new_expand, 2*kernel, mode='same')
    new_expand = scipy.signal.convolve2d(new_expand, np.transpose(2*kernel), mode='same')

    return new_expand


def create_reduce_arr(im, size):
    arr = []
    arr.append(im)
    temp_im = im
    for i in range(size - 1):
        subsampled = temp_im[::2,::2]
        arr.append(subsampled)
        temp_im = subsampled
    return arr


def build_gaussian_pyramid(im, max_levels, filter_size):

    filter_vec = create_gaussian_line(filter_size)

    temp_im = scipy.signal.convolve2d(im, filter_vec, mode='same')
    temp_im = scipy.signal.convolve2d(temp_im, np.transpose(filter_vec), mode='same')

    pyr = create_reduce_arr(temp_im, max_levels)
    pyr[0] = im
    return pyr, filter_vec

def build_laplacian_pyramid(im, max_levels, filter_size):
    pyr = []
    org_reduce, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    # org_reduce = create_reduce_arr(im, max_levels)
    for i in range(max_levels - 1):
        temp_expand = expand(org_reduce[i + 1], filter_vec)
        org_layer = org_reduce[i]
        temp = org_layer - temp_expand
        pyr.append(temp)
    # plt.imshow(org_reduce[-1], cmap='gray')
    # plt.show()
    pyr.append(org_reduce[-1])
    return pyr, filter_vec


def laplacian_to_image(lpyr, filter_vec, coeff):
    pyr_updated = np.multiply(lpyr, coeff)
    cur_layer = lpyr[-1]
    for i in range(len(pyr_updated) - 2, -1, -1):
        cur_layer = expand(cur_layer, filter_vec) + pyr_updated[i]
    return cur_layer


def render_pyramid(pyr, levels):
    positionLst = []
    finalLst = []
    if levels > len(pyr):
        print("error. number of levels to display is more than max_levels")
    width = 0

    for i in range(levels):
        # streching each layer
        pyr[i] = strech_helper(pyr[i])
        width += pyr[i].shape[1]
        positionLst.append((pyr[i].shape[0], pyr[i].shape[1]))

    for i in range(levels):
        zeros = np.zeros(shape=(pyr[0].shape[0], pyr[i].shape[1]))
        zeros[:positionLst[i][0], :positionLst[i][1]] = pyr[i]
        finalLst.append(zeros)
    res = np.concatenate(finalLst, axis=1)

    return res

def strech_helper(im):
    return (im - np.min(im))/(np.max(im) - np.min(im))


def display_pyramid(pyr, levels):
    res = render_pyramid(pyr, levels)
    plt.imshow(res, cmap='gray')
    plt.show()

def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    mask = mask.astype(np.float64)
    lap_pyr1, filter_vec = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    lap_pyr2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)[0]
    gauss_pyr = build_gaussian_pyramid(mask, max_levels, filter_size_mask)[0]
    # TODO: find more elegant way instead of loop
    for i in range(len(gauss_pyr)):
        gauss_pyr[i] = np.array(gauss_pyr[i], dtype=np.float64)
    new_lap_pyr = []
    coeff = [1] * max_levels
    for i in range(max_levels):
        cur_lap_layer = np.multiply(gauss_pyr[i], lap_pyr1[i]) + np.multiply(1 - gauss_pyr[i], lap_pyr2[i])
        new_lap_pyr.append(cur_lap_layer)
    final_image = laplacian_to_image(new_lap_pyr, filter_vec, coeff)
    return np.clip(final_image, 0, 1)


pic = read_image("C:\ex1\gray_orig.png",1)
pic2 = read_image("C:\ex1\/rgb_3_quants.png",1)
lplc = build_laplacian_pyramid(pic, 3, 3)
gauss = build_gaussian_pyramid(pic, 3, 3)

# res = laplacian_to_image(lplc[0], create_gaussian_line(3), [1,1,1])
# new = render_pyramid(lplc[0], 5)
mask = np.ones(shape=(pic.shape[0], pic.shape[1]))
for i in range(int(mask.shape[0]/2)):
    for j in range(int(mask.shape[1]/2)):
        mask[i][j] = 0
blend = pyramid_blending(pic, pic2, mask, 3, 3, 3)

plt.imshow(blend, cmap='gray')
plt.show()