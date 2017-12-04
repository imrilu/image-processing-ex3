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

def blur_spatial(im, kernel_size):
    """
    A method for calculating a blurred version of input picture
    :param im: The image to blur
    :param kernel_size: The size of the gaussian matrix, indicating the intensity of the blur
    :return: The blurred picture
    """
    # assuming kernel_size is odd
    gaussian_kernel = create_gaussian_kernel(kernel_size)
    conv_im = scipy.signal.convolve2d(im, gaussian_kernel, mode='same')
    return conv_im

def create_gaussian_kernel(size):
    """
    A helper method for creating a gaussian kernel with the input size
    :param size: The size of the output dimension of the gaussian kernel
    :return: A discrete gaussian kernel
    """
    bin_arr = np.array([1, 1])
    org_arr = np.array([1, 1])
    sum = 0
    gaussian_matrix = np.zeros(shape=(size, size))
    # TODO: what if size==1 - should return kernel with [1] ?
    if (size == 1):
        # special case, returning a [1] matrix
        return np.array([1])
    for i in range(size-2):
        # iterating to create the initial row of the kernel
        bin_arr = scipy.signal.convolve(bin_arr, org_arr)
    # calculating values on each entry in matrix
    for x in range(size):
        for y in range(size):
            gaussian_matrix[x][y] = bin_arr[x] * bin_arr[y]
            sum += gaussian_matrix[x][y]
    # TODO: search for element-wise multiplication for vector*vector=matrix
    # TODO: maybe create a matrix from repeated row vector
    # normalizing matrix to 1
    for x in range(size):
        for y in range(size):
            gaussian_matrix[x][y] /= sum
    return gaussian_matrix

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

def subsample(im):
    subsampled = im[0::2]
    subsampled = np.transpose(subsampled)
    subsampled = subsampled[0::2]
    subsampled = np.transpose(subsampled)
    # final_sample = np.zeros(shape=(int(im.shape[0]/2), int(im.shape[1]/2)))
    # for i in range(int(im.shape[0]/2)):
    #     final_sample[i] = subsampled[i][0::2]
    # # print(final_sample)
    return subsampled

def expand(im, filter_vec=None):
    new_expand = np.zeros(shape=(int(im.shape[0]*2), int(im.shape[1]*2)))
    # TODO: replace double-loop with padding
    for i in range(int(im.shape[0])):
        for j in range(int(im.shape[1])):
                new_expand[2*i][2*j] = im[i][j]

    if filter_vec is None:
        kernel = create_gaussian_line(3)
    else:
        kernel = filter_vec
    new_expand = scipy.signal.convolve2d(new_expand, kernel, mode='same')
    new_expand = np.transpose(new_expand)
    new_expand = scipy.signal.convolve2d(new_expand, kernel, mode='same')
    new_expand = np.transpose(new_expand)

    return new_expand

def create_reduce_arr(im, size):
    arr = []
    arr.append(im)
    temp_im = im
    for i in range(size):
        subsampled = subsample(temp_im)
        arr.append(subsampled)
        temp_im = subsampled
    return arr


def build_gaussian_pyramid(im, max_levels, filter_size):

    filter_vec = 2 * create_gaussian_line(filter_size)

    temp_im = scipy.signal.convolve2d(im, filter_vec, mode='same')
    temp_im = scipy.signal.convolve2d(temp_im, np.transpose(filter_vec), mode='same')

    pyr = create_reduce_arr(temp_im, max_levels)

    return filter_vec, pyr

def build_laplacian_pyramid(im, max_levels, filter_size):
    pyr = []
    filter_vec, org_reduce = build_gaussian_pyramid(im, max_levels, filter_size)
    # org_reduce = create_reduce_arr(im, max_levels)
    for i in range(len(org_reduce) - 1):
        temp_expand = expand(org_reduce[i + 1], filter_vec)
        org_layer = org_reduce[i]
        temp = org_layer - temp_expand
        pyr.append(temp)
    pyr.append(org_reduce[-1])
    return filter_vec, pyr

def laplacian_to_image(lpyr, filter_vec, coeff):
    filteredVector = 2*filter_vec
    pyr_updated = np.multiply(lpyr, coeff)
    cur_layer = lpyr[-1]
    for i in range(len(pyr_updated) - 2, -1, -1):
        cur_layer = expand(cur_layer, filteredVector) + pyr_updated[i]
    return cur_layer

def render_pyramid(pyr, levels):
    positionLst = []
    finalLst = []
    if levels > len(pyr):
        print("error. number of levels to display is more than max_levels")
    width = 0
    for i in range(levels):
        width += pyr[i].shape[1]
        positionLst.append((pyr[i].shape[0], pyr[i].shape[1]))

    for i in range(levels):
        zeros = np.zeros(shape=(pyr[0].shape[0], pyr[i].shape[1]))
        zeros[:positionLst[i][0], :positionLst[i][1]] = pyr[i]
        finalLst.append(zeros)
    res = np.concatenate(finalLst, axis=1)

    return res


def display_pyramid(pyr, levels):
    res = render_pyramid(pyr, levels)
    plt.imshow(res, cmap='gray')
    plt.show()

def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    mask = mask.astype(double)
    filter_vec, lap_pyr1 = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    lap_pyr2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)[1]
    gauss_pyr = build_gaussian_pyramid(mask, max_levels, filter_size_mask)[1].astype(np.float64)
    new_lap_pyr = []
    coeff = [1] * max_levels
    for i in range(max_levels):
        cur_lap_layer = np.multiply(gauss_pyr[i], lap_pyr1[i]) + np.multiply(1 - gauss_pyr[i], lap_pyr2[i])
        new_lap_pyr.append(cur_lap_layer)
    final_image = laplacian_to_image(new_lap_pyr, filter_vec, coeff)
    return np.clip(final_image, 0, 1)

def strech_helper(image, hist):
    """
    A helper function for stretching the image
    :param image: The input picture to be stretched
    :param hist: The histogram of the input image
    :return: The stretched image
    """
    image = (image * 255).astype(int)
    cdf = np.cumsum(hist)
    cdf = 255 * cdf / cdf[-1]
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    return (cdf_m[image] / 255).astype(np.float64)

pic = read_image("C:\ex1\gray_orig.png",1)
lplc = build_laplacian_pyramid(pic, 4, 3)
gauss = build_laplacian_pyramid(pic, 4, 3)

res = render_pyramid(lplc[1], 4)

plt.imshow(res, cmap='gray')
plt.show()