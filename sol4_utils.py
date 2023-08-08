from scipy.signal import convolve2d
import numpy as np
from imageio import imread
from skimage.color import rgb2gray
from scipy.ndimage.filters import convolve

NORMALIZED_FACTOR = 255
GRAYSCALE = 1
CONVOLUTION_BASIC_ARRAY = [1, 1]
BLUR_FACTOR = 2
LOWEST_DIMENSION = 16



def gaussian_kernel(kernel_size):
    conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
    conv_kernel = convolve2d(conv_kernel, conv_kernel.T)
    kernel = np.array([1], dtype=np.float64)[:, None]
    for i in range(kernel_size - 1):
        kernel = convolve2d(kernel, conv_kernel, 'full')
    return kernel / kernel.sum()


def blur_spatial(img, kernel_size):
    kernel = gaussian_kernel(kernel_size)
    blur_img = np.zeros_like(img)
    if len(img.shape) == 2:
        blur_img = convolve2d(img, kernel, 'same', 'symm')
    else:
        for i in range(3):
            blur_img[..., i] = convolve2d(img[..., i], kernel, 'same', 'symm')
    return blur_img

# Function from the previous exercise:
def read_image(filename, representation):
    """
    Reads an image and converts it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    :return: Returns the image as an np.float64 matrix normalized to [0,1]
    """
    image = imread(filename)
    image = image.astype(np.float64)
    image /= NORMALIZED_FACTOR
    if representation == GRAYSCALE:
        return rgb2gray(image)
    return image

# Function from the previous exercise:
def reduce(im, blur_filter):
    """
    Reduces an image by a factor of 2 using the blur filter
    :param im: Original image
    :param blur_filter: Blur filter
    :return: the downsampled image
    """
    # Blur:
    downsampled_image = convolve(im, blur_filter, mode='wrap')
    cols_blur_filter = np.transpose(blur_filter)
    downsampled_image = convolve(downsampled_image, cols_blur_filter, mode='wrap')
    # Sub-sample (select only every second pixel in every second row):
    return downsampled_image[::BLUR_FACTOR, ::BLUR_FACTOR]

def is_minimum_dimension_reached(image):
    """ Answers to the question: Is the image dimension (height or width) is smaller than 16?
    :param im: a grayscale image.
    :return: True if the image dimension (height or width) is smaller than 16."""
    if image.shape[0] < LOWEST_DIMENSION:
        return True
    elif image.shape[1] < LOWEST_DIMENSION:
        return True
    return False

def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    Builds a gaussian pyramid for a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter
            (an odd scalar that represents a squared filter)
            to be used in constructing the pyramid filter
    :return: pyr, filter_vec. Where pyr is the resulting pyramid as a
            standard python array with maximum length of max_levels,
            where each element of the array is a grayscale image.
            and filter_vec is a row vector of shape (1, filter_size)
            used for the pyramid construction.
    """
    # Compute filter vector:
    temp_vec = np.convolve([1], CONVOLUTION_BASIC_ARRAY)
    for i in range(1, filter_size - 1):  # assume the filter size will be >=2
        temp_vec = np.convolve(temp_vec, CONVOLUTION_BASIC_ARRAY)
    filter_vec_shape = (1, filter_size)
    normalization_factor = 1 / (pow(BLUR_FACTOR, filter_size - 1))
    temp_vec = temp_vec.reshape(filter_vec_shape)
    filter_vec = temp_vec * normalization_factor
    # Build the gaussian pyramid:
    pyr = [im]
    for i in range(1, max_levels):
        pyr.append(reduce(pyr[-1], filter_vec))
        if is_minimum_dimension_reached(pyr[-1]):
            return pyr[:-1], filter_vec
    return pyr, filter_vec
