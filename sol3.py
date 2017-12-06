import numpy as np
import scipy
import scipy.signal
from skimage.color import rgb2gray
from scipy.misc import imread as imread
from scipy import linalg as linalg
import matplotlib.pyplot as plt
import os

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
    A helper method for creating a gaussian kernel 'line' with the input size
    :param size: The size of the output dimension of the gaussian kernel
    :return: A discrete gaussian kernel
    """
    bin_arr = np.array([1, 1])
    org_arr = np.array([1, 1])
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
    """
    a helper method for expanding the image by double from it's input size
    :param im: the input picture to expand
    :param filter_vec: a custom filter in case we'd like to convolve with different one
    :return: the expanded picture after convolution
    """
    new_expand = np.zeros(shape=(int(im.shape[0]*2), int(im.shape[1]*2)))
    new_expand[::2,::2] = im
    if filter_vec is None:
        kernel = create_gaussian_line(3)
    else:
        kernel = filter_vec
    new_expand = scipy.signal.convolve2d(new_expand, 2*kernel, mode='same')
    new_expand = scipy.signal.convolve2d(new_expand, np.transpose(2*kernel), mode='same')

    return new_expand


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    a method for building a gaussian pyramid
    :param im: the input image to construct the pyramid from
    :param max_levels: maximum levels in the pyramid
    :param filter_size: the size of the gaussian filter we're using
    :return: an array representing the pyramid
    """
    filter_vec = create_gaussian_line(filter_size)
    # creating duplicate for confy use
    temp_im = im
    pyr = [im]


    for i in range(max_levels - 1):
        # blurring the cur layer
        temp_im = scipy.signal.convolve2d(temp_im, filter_vec, mode='same')
        temp_im = scipy.signal.convolve2d(temp_im, np.transpose(filter_vec), mode='same')
        # sampling only every 2nd row and column
        temp_im = temp_im[::2, ::2]
        pyr.append(temp_im)

    return pyr, filter_vec

def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    a method for building a laplacian pyramid
    :param im: the input image to construct the pyramid from
    :param max_levels: maximum levels in the pyramid
    :param filter_size: the size of the laplacian filter we're using
    :return: an array representing the pyramid
    """
    pyr = []
    org_reduce, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
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
    """
    a method that constructs the original image from it's laplacian pyramid
    :param lpyr: the laplacian pyramid of the image we'd like to construct
    :param filter_vec: the filter vector to be used in the reconstruction
    :param coeff: the coefficients for each layer of the pyramid
    :return: the reconstructed image
    """
    pyr_updated = np.multiply(lpyr, coeff)
    cur_layer = lpyr[-1]
    for i in range(len(pyr_updated) - 2, -1, -1):
        cur_layer = expand(cur_layer, filter_vec) + pyr_updated[i]
    return cur_layer


def render_pyramid(pyr, levels):
    """
    render the pyramid and construct a single image representing all the layers horizontally
    :param pyr: the image's pyramid
    :param levels: the number of levels of the pyramid
    :return: an image representing all the pyramid's layers horizontally
    """
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
    """
    helper function for streching and equalizing the image between 0 and 1
    :param im: input picture to equalize
    :return: equalized picture between 0 and 1
    """
    return (im - np.min(im))/(np.max(im) - np.min(im))


def display_pyramid(pyr, levels):
    """
    displaying the pyramid into the screen using plt
    :param pyr: the input pyramid to be displayed
    :param levels: number of levels of the pyramid
    """
    res = render_pyramid(pyr, levels)
    plt.imshow(res, cmap='gray')
    plt.show()

def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    a method for blending 2 pictures using a binary mask
    :param im1: the first picture to blend
    :param im2: the second picture to blend
    :param mask: the binary mask
    :param max_levels: number of max levels to be used while constructing the pyramids
    :param filter_size_im: size of the filter for the images
    :param filter_size_mask: size of the filter for the mask
    :return:
    """
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

def relpath(filename):
    """
    helper method for using relative paths to load the pictures
    :param filename: the relative path to be parsed
    :return: the real path
    """
    return os.path.join(os.path.dirname(__file__), filename)

def blending_example1():
    """
    a method for creating a blending example constructing a blend from 2 rgb images
    :return: the blended picture
    """
    pic_desert = read_image(relpath(".\externals\pic_desert.jpg"), 2)
    pic_pool = read_image(relpath(".\externals\pic_swim.jpg"), 2)
    mask = read_image(relpath(".\externals\mask_desert.jpg"), 1)
    # making the mask binary (normalizing 2 original values)
    mask = strech_helper(mask)
    [R1, G1, B1] = np.dsplit(pic_desert, pic_desert.shape[2])
    [R2, G2, B2] = np.dsplit(pic_pool, pic_pool.shape[2])
    R1 = np.reshape(R1, (512,1024))
    R2 = np.reshape(R2, (512,1024))
    G1 = np.reshape(G1, (512,1024))
    G2 = np.reshape(G2, (512,1024))
    B1 = np.reshape(B1, (512,1024))
    B2 = np.reshape(B2, (512,1024))

    blend1 = pyramid_blending(R2, R1, mask, 3, 3, 3)
    blend2 = pyramid_blending(G2, G1, mask, 3, 3, 3)
    blend3 = pyramid_blending(B2, B1, mask, 3, 3, 3)

    blend1 = np.reshape(blend1, (blend1.shape[0], blend1.shape[1], 1))
    blend2 = np.reshape(blend2, (blend2.shape[0], blend3.shape[1], 1))
    blend3 = np.reshape(blend3, (blend3.shape[0], blend3.shape[1], 1))

    new_pic = np.concatenate((blend1, blend2, blend3), axis=2)

    plt.imshow(new_pic, cmap='gray')
    plt.show()

    return pic_desert, pic_pool, mask, new_pic

def blending_example2():
    """
    a method for creating a blending example constructing a blend from 2 rgb images
    :return: the blended picture
    """
    pic_earth = read_image(relpath(".\externals\pic_earth.jpg"), 2)
    pic_asteroid = read_image(relpath(".\externals\pic_asteroid.jpg"), 2)
    mask = read_image(relpath(".\externals\mask_asteroid.jpg"), 1)
    # making the mask binary (normalizing 2 original values)
    mask = strech_helper(mask)
    [R1, G1, B1] = np.dsplit(pic_earth, pic_earth.shape[2])
    [R2, G2, B2] = np.dsplit(pic_asteroid, pic_asteroid.shape[2])
    R1 = np.reshape(R1, (1024,1024))
    R2 = np.reshape(R2, (1024,1024))
    G1 = np.reshape(G1, (1024,1024))
    G2 = np.reshape(G2, (1024,1024))
    B1 = np.reshape(B1, (1024,1024))
    B2 = np.reshape(B2, (1024,1024))

    blend1 = pyramid_blending(R2, R1, mask, 3, 3, 3)
    blend2 = pyramid_blending(G2, G1, mask, 3, 3, 3)
    blend3 = pyramid_blending(B2, B1, mask, 3, 3, 3)

    blend1 = np.reshape(blend1, (blend1.shape[0], blend1.shape[1], 1))
    blend2 = np.reshape(blend2, (blend2.shape[0], blend3.shape[1], 1))
    blend3 = np.reshape(blend3, (blend3.shape[0], blend3.shape[1], 1))

    new_pic = np.concatenate((blend1, blend2, blend3), axis=2)

    plt.imshow(new_pic, cmap='gray')
    plt.show()

    return pic_earth, pic_asteroid, mask, new_pic


# pic = read_image("C:\ex1\gray_orig.png",1)
# pic2 = read_image("C:\ex1\/rgb_3_quants.png",1)
# lplc = build_laplacian_pyramid(pic, 5, 3)
# gauss = build_gaussian_pyramid(pic, 5, 3)
#
# res = laplacian_to_image(lplc[0], create_gaussian_line(3), [1] * 5)
# new = render_pyramid(lplc[0], 5)
# mask = np.ones(shape=(pic.shape[0], pic.shape[1]))
# for i in range(int(mask.shape[0]/2)):
#     for j in range(int(mask.shape[1]/2)):
#         mask[i][j] = 0
# blend = pyramid_blending(pic, pic2, mask, 3, 3, 3)

# for i in range(len(gauss[0])):
#     plt.imshow(gauss[0][i], cmap='gray')
#     plt.figure()

# plt.imshow(new, cmap='gray')
# plt.show()
blending_example2()