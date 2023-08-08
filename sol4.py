# Initial code for ex4.
# You may change this code, but keep the functions' signatures
# You can also split the code to multiple files as long as this file's API is unchanged


import numpy as np
import os
import matplotlib.pyplot as plt

from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import label, center_of_mass, convolve, map_coordinates
import shutil
from imageio import imwrite

import sol4_utils

KERNEL_SIZE = 3
K_RESPONSE_MATRIX = 0.04
PATCH_SIZE = 7
MINIMAL_DIST = 13
DESCRIPTOR_RADIUS = 3
CONVERT_FIRST_THIRD_LEVELS = 0.25
SUBSCRIPT_EINSUM = 'ij, kj->ki'
HORIZONTAL_AXIS = 1
VERTICAL_AXIS = 0
SECOND_FROM_END_INDEX = -2
TRANSLATION_NUM_OF_PAIRS = 1
RIGID_NUM_OF_PAIRS = 2
RED_COLOR = 'r'
YELLOW_COLOR = 'y'
BLUE_COLOR = 'b'


def harris_corner_detector(im):
  """
  Detects harris corners.
  Make sure the returned coordinates are x major!!!
  :param im: A 2D array representing an image.
  :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.  
  """
  # Get Ix derivative of the image using the filter [1, 0, −1]:
  derivative_x_filter = np.asarray([1, 0, -1]).reshape(1, 3)
  Ix = convolve(im, derivative_x_filter)

  # Get Iy derivative of the image using the filters [1, 0, −1]^T:
  derivative_y_filter = np.transpose(derivative_x_filter)
  Iy = convolve(im, derivative_y_filter)

  # Blur the images: Ix^2, Iy^2, IxIy using blur_spatial function with KERNEL_SIZE:
  Ix_mult_Ix = sol4_utils.blur_spatial(Ix * Ix, KERNEL_SIZE)
  Iy_mult_Iy = sol4_utils.blur_spatial(Iy * Iy, KERNEL_SIZE)
  Ix_mult_Iy = sol4_utils.blur_spatial(Ix * Iy, KERNEL_SIZE)
  Iy_mult_Ix = sol4_utils.blur_spatial(Iy * Ix, KERNEL_SIZE)

  # Compute R = det(M) − k(trace(M))^2:
  det_M = (Ix_mult_Ix * Iy_mult_Iy) - (Ix_mult_Iy * Iy_mult_Ix)
  trace_M = Ix_mult_Ix + Iy_mult_Iy
  R = det_M - (K_RESPONSE_MATRIX * (trace_M * trace_M))

  # To find corner points use non_maximum_suppression function:
  local_maximum_points = non_maximum_suppression(R)

  # Return the xy coordinates of the corners:
  # Note: coordinate order is (x,y), as opposed to the order when indexing an array which is (row,column)
  return np.fliplr(np.argwhere(local_maximum_points))


def create_coordinates(pos_x, pos_y, desc_rad):
  """
  Create coordinates array of the patch at level 3 of the pyramid.
  around point (pos_x, pos_y).
  :param pos_x: x coordinate of point p
  :param pos_y: y coordinate of point p
  :param desc_rad: "Radius" of descriptors to compute.
  :return: coordinates array of the patch for the point (pos_x, pos_y).
  """
  start = -desc_rad
  end = desc_rad + 1
  lst = []
  for i in range(start, end):
    for j in range(start, end):
      lst.append((pos_y + i, pos_x + j))
  return np.transpose(np.array(lst))

def normalize_descriptor(d):
  """
  Normalize the  descriptor matrix such that returned_d = (d − µ)/||d − µ||:
  in cases where the norm is zero return the zero descriptor
  :param d: A 2D array with shape (K,K) descriptor
  :return: A 2D array with shape (K,K) represents normalize descriptor
  """
  d_minus_mean = d - np.mean(d) # d − µ
  norm_d_minus_mean = np.linalg.norm(d_minus_mean) #||d − µ||
  if norm_d_minus_mean == 0:
    return d_minus_mean # norm is zero return the zero descriptor
  return d_minus_mean/norm_d_minus_mean # (d − µ)/||d − µ||


def compute_descriptor(pos_x, pos_y, desc_rad, im, descriptor_size):
  """
  Compute descriptor of a point (pos_x, pos_y).
  :param pos_x: x coordinate of point p
  :param pos_y: y coordinate of point p
  :param desc_rad: "Radius" of descriptors to compute.
  :param im: A 2D array representing an image.
  :param descriptor_size: size of the descriptor.
  :return: descriptor of a point (pos_x, pos_y)
  """
  coordinates = create_coordinates(pos_x, pos_y, desc_rad)
  # Sample them at these sub-pixel coordinates interpolating within the pyramid’s 3rd level image properly
  d = map_coordinates(input=im, coordinates=coordinates, order=1, prefilter=False)
  d_shape = (descriptor_size, descriptor_size)
  return normalize_descriptor(np.reshape(d, d_shape))



def sample_descriptor(im, pos, desc_rad):
  """
  Samples descriptors at the given corners.
  :param im: A 2D array representing an image.
  :param pos: An array with shape (N,2), where pos[i,:] are the [x,y] coordinates of the ith corner point.
  :param desc_rad: "Radius" of descriptors to compute.
  :return: A 3D array with shape (N,K,K) containing the ith descriptor at desc[i,:,:].
  """
  # Find N, K:
  K = 1 + (2 * desc_rad)
  N = pos.shape[0]  # len(pos[:, 0])
  descriptors_array = np.zeros((N, K, K))
  # Note:  the grayscale image im to be the 3rd level pyramid image
  for i in range(0,N):
    descriptors_array[i, :, :] = \
      compute_descriptor(pos[i][0], pos[i][1], desc_rad, im, K)
  return descriptors_array


def find_features(pyr):
  """
  Detects and extracts feature points from a pyramid.
  :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
  :return: A list containing:
              1) An array with shape (N,2) of [x,y] feature location per row found in the image.
                 These coordinates are provided at the pyramid level pyr[0].
              2) A feature descriptor array with shape (N,K,K)
  """
  # Get the keypoints:
  keypoints = spread_out_corners(pyr[0], PATCH_SIZE, PATCH_SIZE, MINIMAL_DIST)
  # Sampling a descriptor for each keypoint:
  pos = keypoints.astype(np.float64) * CONVERT_FIRST_THIRD_LEVELS
  sample_descriptors = sample_descriptor(pyr[2], pos, DESCRIPTOR_RADIUS)
  lst = []
  lst.append(keypoints)
  lst.append(sample_descriptors)
  return lst


def find_match_descriptors(S, horizontal_second_max, vertical_second_max, min_score):
  """
  Finds indices of match descriptors.Tow descriptors define as match if the three properties below hold:
  (*) Sj,k ≥ 2ndmax{Sj,l |l ∈ {0..f1 −1}}
  (*) Sj,k ≥ 2ndmax{Sl,k |l ∈ {0..f0 − 1}}
  (*) Sj,k is greater (>) than some predefined minimal score
  :param S: Array represents the scores shape (N1,N2).
  :param horizontal_second_max: Array represents the2nd max scores shape(N1, 1)
  :param vertical_second_max:  Array represents the2nd max scores shape(1, N2)
  :param min_score: Minimal match score.
  :return: Indices of match descriptors
  """
  # First property Sj,k ≥ 2ndmax{Sj,l |l ∈ {0..f1 −1}}:
  first_property = (S >= vertical_second_max)

  # Second property Sj,k ≥ 2ndmax{Sl,k |l ∈ {0..f0 − 1}}:
  second_property = (S >= horizontal_second_max)

  # Third property Sj,k is greater (>) than some predefined minimal score:
  third_property = (S > min_score)

  match_descriptors_indices = np.argwhere(first_property & second_property & third_property)
  return np.fliplr(match_descriptors_indices)


def match_features(desc1, desc2, min_score):
  """
  Return indices of matching descriptors.
  :param desc1: A feature descriptor array with shape (N1,K,K).
  :param desc2: A feature descriptor array with shape (N2,K,K).
  :param min_score: Minimal match score.
  :return: A list containing:
              1) An array with shape (M,) and dtype int of matching indices in desc1.
              2) An array with shape (M,) and dtype int of matching indices in desc2.
  """
  # The match score between two descriptors will simply be their dot product (flattened to 1D arrays).
  flattened_desc1 = desc1[:, np.newaxis]
  flattened_desc2 = desc2[np.newaxis, :]
  dot_product = np.multiply(flattened_desc1, flattened_desc2)
  # S[j,k] = D[i,j]·D[i+1,k] the match score between the jth descriptor in frame i and the kth descriptor in frame i+1
  sum_axes = (2, 3)  # K rows and K cols
  S = dot_product.sum(sum_axes)
  S_horizontal_sorted = np.partition(S, SECOND_FROM_END_INDEX, HORIZONTAL_AXIS)
  horizontal_second_max = S_horizontal_sorted[:, SECOND_FROM_END_INDEX][:, np.newaxis]
  S_vertical_sorted = np.partition(S, SECOND_FROM_END_INDEX, VERTICAL_AXIS)
  vertical_second_max = S_vertical_sorted[SECOND_FROM_END_INDEX][np.newaxis, :]
  lst = []
  match_descriptors_indexes = find_match_descriptors(S, horizontal_second_max, vertical_second_max, min_score)
  matching_indices_in_desc1 = match_descriptors_indexes[:, HORIZONTAL_AXIS]
  lst.append(matching_indices_in_desc1)
  matching_indices_in_desc2 = match_descriptors_indexes[:, VERTICAL_AXIS]
  lst.append(matching_indices_in_desc2)
  return lst


def apply_homography(pos1, H12):
  """
  Apply homography to inhomogenous points.
  :param pos1: An array with shape (N,2) of [x,y] point coordinates.
  :param H12: A 3x3 homography matrix.
  :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12.
  """
  # Transform each point (x1,y1) to (x1,y1,1):
  N = pos1.shape[0]
  ones_coordinates = np.ones((N, 1))
  homogenous_pos1 = np.hstack((pos1, ones_coordinates))
  # multiply H1,2 and each point (x1,y1,1):
  H12_mult_homogenous_pos1 = np.einsum(SUBSCRIPT_EINSUM, H12, homogenous_pos1)
  # divide each coordinate of point in z coordinate:
  z_coordinates = H12_mult_homogenous_pos1[:, 2]
  x_coordinates = H12_mult_homogenous_pos1[:, 0] / z_coordinates
  y_coordinates = H12_mult_homogenous_pos1[:, 1] / z_coordinates
  transformed_pos1 = np.transpose(np.vstack((x_coordinates, y_coordinates)))
  return transformed_pos1


def preform_ransac_iteration(num_of_pairs, points1, points2, inlier_tol, translation_only):
  """
  Preforms a RANSAC iteration, returns inliers set from the current iteration.
  :param num_of_pairs: num pairs of points are needed in each ransac iteration.
  :param points1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
  :param points2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
  :param inlier_tol: inlier tolerance threshold.
  :param translation_only:  see estimate rigid transform
  :return: inliers set from the current iteration.
  """
  # Pick a random set of 2 point matches from the supplied N point matches:
  N = points2.shape[0]
  random_indices = np.arange(N)
  np.random.shuffle(random_indices)
  point1 = points1[random_indices[:num_of_pairs], :]
  point2 = points2[random_indices[:num_of_pairs], :]

  # Compute the homography H1,2 that transforms the 2 points P1,J to the 2 points P2,J:
  H12 = estimate_rigid_transform(point1, point2, translation_only)

  # Compute the squared euclidean distance Ej = ||P′[2,j] − P[2,j]||^2 for j = 0..N − 1
  points2_tag = apply_homography(points1, H12)
  E = (np.linalg.norm(points2_tag - points2, None, HORIZONTAL_AXIS)) **2

  # Mark all matches having Ej < inlier_tol as inlier matches and the rest as outlier matches
  return np.arange(N)[E < inlier_tol]


def ransac_homography(points1, points2, num_iter, inlier_tol, translation_only=False):
  """
  Computes homography between two sets of points using RANSAC.
  :param pos1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
  :param pos2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
  :param num_iter: Number of RANSAC iterations to perform.
  :param inlier_tol: inlier tolerance threshold.
  :param translation_only: see estimate rigid transform
  :return: A list containing:
              1) A 3x3 normalized homography matrix.
              2) An Array with shape (S,) where S is the number of inliers,
                  containing the indices in pos1/pos2 of the maximal set of inlier matches found.
  """
  lst = []
  largest_set_of_inliers = np.array([])
  largest_set_size = 0
  num_of_pairs = RIGID_NUM_OF_PAIRS
  # In case (translation only = True) need one pair of points in each ransac iteration.
  if translation_only == True:
    num_of_pairs = TRANSLATION_NUM_OF_PAIRS
  # RANSAC performs several iterations of 3 steps:
  for i in range(0, num_iter):
    inliers_result = preform_ransac_iteration(num_of_pairs, points1, points2, inlier_tol, translation_only)
    # keeping a record of the largest set of inliers Jin it has come upon:
    if inliers_result.size > largest_set_size:
      largest_set_of_inliers = inliers_result
      largest_set_size = inliers_result.size

  # return estimate_rigid_transform on these inlier point matches P1,Jin and P2,Jin
  computed_homography = estimate_rigid_transform(points1[largest_set_of_inliers, :],
                                                 points2[largest_set_of_inliers, :],
                                                translation_only)
  lst.append(computed_homography)
  lst.append(largest_set_of_inliers)
  return lst



def display_matches(im1, im2, points1, points2, inliers):
  """
  Display matching points.
  :param im1: A grayscale image.
  :param im2: A grayscale image.
  :parma pos1: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
  :param pos2: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
  :param inliers: An array with shape (S,) of inlier matches.
  """
  # Use np.hstack of an image pair im1 and im2, with the matched points provided in pos1 and pos2
  # overlayed correspondingly as red dots:
  concatenated_image = np.hstack((im1, im2))
  plt.imshow(concatenated_image, 'gray')

  # Each outlier match, say at index j, is denoted by plotting a blue line between
  # pos1[j,:] and the horizontally shifted pos2[j,:]:
  outlier_x_im1 = points1[:, 0]
  outlier_x_im2 = points2[:, 0] + im1.shape[1]  # Shifting pos2:
  outlier_x_coordinates = [outlier_x_im1, outlier_x_im2]
  outlier_y_im1 = points1[:, 1]
  outlier_y_im2 = points2[:, 1]
  outlier_y_coordinates = [outlier_y_im1, outlier_y_im2]
  plt.plot(outlier_x_coordinates, outlier_y_coordinates, mfc=RED_COLOR, mec=RED_COLOR, c=BLUE_COLOR, lw=.2, ms=1,
           marker='o')

  # Inlier matches are denoted by plotting a yellow line between the matched points:
  inlier_x_im1 = points1[:, 0][inliers]
  inlier_x_im2 = points2[:, 0][inliers] + + im1.shape[1]  # Shifting pos2:
  inlier_x_coordinates = [inlier_x_im1, inlier_x_im2]
  inlier_y_im1 = points1[:, 1][inliers]
  inlier_y_im2 = points2[:, 1][inliers]
  inlier_y_coordinates = [inlier_y_im1, inlier_y_im2]
  plt.plot(inlier_x_coordinates, inlier_y_coordinates, mfc=RED_COLOR, mec=RED_COLOR, c=YELLOW_COLOR, lw=.4, ms=1,
           marker='o')
  plt.show()
  return


def accumulate_homographies(H_succesive, m):
  """
  Convert a list of succesive homographies to a
  list of homographies to a common reference frame.
  :param H_successive: A list of M-1 3x3 homography
    matrices where H_successive[i] is a homography which transforms points
    from coordinate system i to coordinate system i+1.
  :param m: Index of the coordinate system towards which we would like to
    accumulate the given homographies.
  :return: A list of M 3x3 homography matrices,
    where H2m[i] transforms points from coordinate system i to coordinate system m
  """
  M = len(H_succesive) + 1  # because H_successive is list of M-1 3x3 homography
  H_tag_succesive_shape = (M, 3, 3)
  H_tag_succesive = np.zeros(H_tag_succesive_shape)

  # For i = m we set H¯[i,m] to the 3 × 3 identity matrix I = np.eye(3):
  H_tag_succesive[m] = np.eye(3)

  # For i < m we set H¯[i,m] = H[m−1,m] ∗ ... ∗ H[i+1,i+2] ∗ H[i,i+1] :
  for i in range(m-1, -1, -1):
    H_tag_succesive[i] = np.dot(H_tag_succesive[i+1],H_succesive[i])
    # Normalize:
    H_tag_succesive[i] /= (H_tag_succesive[i][2,2])

  # For i > m we set H¯[i,m] = H^−1[m,m+1] ∗ ... ∗ H^−1[i−2,i−1]∗ H^−1[i−1,i] :
  for i in range(m + 1, M):
    inversed_last_H = np.linalg.inv(H_succesive[i - 1])
    H_tag_succesive[i] = np.dot(H_tag_succesive[i - 1], inversed_last_H)
    # Normalize:
    H_tag_succesive[i] /= (H_tag_succesive[i][2, 2])

  lst = list(H_tag_succesive)
  return lst


def compute_bounding_box(homography, w, h):
  """
  computes bounding box of warped image under homography, without actually warping the image
  :param homography: homography
  :param w: width of the image
  :param h: height of the image
  :return: 2x2 array, where the first row is [x,y] of the top left corner,
   and the second row is the [x,y] of the bottom right corner
  """
  # Compute where the 4 corner pixel coordinates of each frame Ii get mapped to by H¯[i,m]:
  #(top - left, top - right, bottom - right,bottom - left)
  t_left = [0, 0]
  t_right = [w - 1, 0]
  b_right = [w - 1, h - 1]
  b_left =  [0, h - 1]
  positions = np.array([t_left, t_right, b_left, b_right])
  panorama_coordinate = apply_homography(pos1=positions, H12=homography)

  # Define the region in which the panorama image Ipano should be rendered denoted xmax, xmin, ymax, ymin:
  x_min = np.min(panorama_coordinate[:, VERTICAL_AXIS])
  y_min = np.min(panorama_coordinate[:, HORIZONTAL_AXIS])
  top_left_corner = [x_min, y_min]

  x_max = np.max(panorama_coordinate[:, VERTICAL_AXIS])
  y_max = np.max(panorama_coordinate[:, HORIZONTAL_AXIS])
  bottom_right_corner = [x_max, y_max]

  return np.array([top_left_corner, bottom_right_corner], dtype=np.int)


def warp_channel(image, homography):
  """
  Warps a 2D image with a given homography.
  :param image: a 2D image.
  :param homography: homograhpy.
  :return: A 2d warped image.
  """
  # divide the panorama to M vertical strips, each covering a portion of the full lateral range [xmin, xmax]
  bounding_box = compute_bounding_box(homography, image.shape[1], image.shape[0])

  # Back-warping of the strips should then be performed:
  # Using the function np.meshgrid to hold the x and y coordinates of each of the warped image:
  x_min = bounding_box[0][0]
  x_max = bounding_box[1][0]
  y_min = bounding_box[0][1]
  y_max = bounding_box[1][1]
  x_coordinates, y_coordinates = np.meshgrid(np.arange(x_min, x_max+1), np.arange(y_min, y_max+1))
  # Transformed by the inverse homography H¯^−1 i,m using apply_homography back to the coordinate system:
  positions = np.dstack((x_coordinates, y_coordinates))
  positions = positions.reshape(x_coordinates.shape[0] * x_coordinates.shape[1], 2)
  inverse_homography = np.linalg.inv(homography)
  new_positions = apply_homography(positions, inverse_homography).reshape(x_coordinates.shape[0],
                                                                          x_coordinates.shape[1],2)
  # Interpolate the image with map_coordinates
  coordinates = [new_positions[:, :, HORIZONTAL_AXIS], new_positions[:, :, VERTICAL_AXIS]]
  return map_coordinates(input=image, coordinates=coordinates, order=1, prefilter=False)


def warp_image(image, homography):
  """
  Warps an RGB image with a given homography.
  :param image: an RGB image.
  :param homography: homograhpy.
  :return: A warped image.
  """
  return np.dstack([warp_channel(image[...,channel], homography) for channel in range(3)])


def filter_homographies_with_translation(homographies, minimum_right_translation):
  """
  Filters rigid transformations encoded as homographies by the amount of translation from left to right.
  :param homographies: homograhpies to filter.
  :param minimum_right_translation: amount of translation below which the transformation is discarded.
  :return: filtered homographies..
  """
  translation_over_thresh = [0]
  last = homographies[0][0,-1]
  for i in range(1, len(homographies)):
    if homographies[i][0,-1] - last > minimum_right_translation:
      translation_over_thresh.append(i)
      last = homographies[i][0,-1]
  return np.array(translation_over_thresh).astype(np.int)


def estimate_rigid_transform(points1, points2, translation_only=False):
  """
  Computes rigid transforming points1 towards points2, using least squares method.
  points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate is *x*.
  :param points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
  :param points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
  :param translation_only: whether to compute translation only. False (default) to compute rotation as well.
  :return: A 3x3 array with the computed homography.
  """
  centroid1 = points1.mean(axis=0)
  centroid2 = points2.mean(axis=0)

  if translation_only:
    rotation = np.eye(2)
    translation = centroid2 - centroid1

  else:
    centered_points1 = points1 - centroid1
    centered_points2 = points2 - centroid2

    sigma = centered_points2.T @ centered_points1
    U, _, Vt = np.linalg.svd(sigma)

    rotation = U @ Vt
    translation = -rotation @ centroid1 + centroid2

  H = np.eye(3)
  H[:2,:2] = rotation
  H[:2, 2] = translation
  return H


def non_maximum_suppression(image):
  """
  Finds local maximas of an image.
  :param image: A 2D array representing an image.
  :return: A boolean array with the same shape as the input image, where True indicates local maximum.
  """
  # Find local maximas.
  neighborhood = generate_binary_structure(2,2)
  local_max = maximum_filter(image, footprint=neighborhood)==image
  local_max[image<(image.max()*0.1)] = False

  # Erode areas to single points.
  lbs, num = label(local_max)
  centers = center_of_mass(local_max, lbs, np.arange(num)+1)
  centers = np.stack(centers).round().astype(np.int)
  ret = np.zeros_like(image, dtype=np.bool)
  ret[centers[:,0], centers[:,1]] = True

  return ret


def spread_out_corners(im, m, n, radius):
  """
  Splits the image im to m by n rectangles and uses harris_corner_detector on each.
  :param im: A 2D array representing an image.
  :param m: Vertical number of rectangles.
  :param n: Horizontal number of rectangles.
  :param radius: Minimal distance of corner points from the boundary of the image.
  :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
  """
  corners = [np.empty((0,2), dtype=np.int)]
  x_bound = np.linspace(0, im.shape[1], n+1, dtype=np.int)
  y_bound = np.linspace(0, im.shape[0], m+1, dtype=np.int)
  for i in range(n):
    for j in range(m):
      # Use Harris detector on every sub image.
      sub_im = im[y_bound[j]:y_bound[j+1], x_bound[i]:x_bound[i+1]]
      sub_corners = harris_corner_detector(sub_im)
      sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis,:]
      corners.append(sub_corners)
  corners = np.vstack(corners)
  legit = ((corners[:,0]>radius) & (corners[:,0]<im.shape[1]-radius) &
           (corners[:,1]>radius) & (corners[:,1]<im.shape[0]-radius))
  ret = corners[legit,:]
  return ret


class PanoramicVideoGenerator:
  """
  Generates panorama from a set of images.
  """

  def __init__(self, data_dir, file_prefix, num_images, bonus=False):
    """
    The naming convention for a sequence of images is file_prefixN.jpg,
    where N is a running number 001, 002, 003...
    :param data_dir: path to input images.
    :param file_prefix: see above.
    :param num_images: number of images to produce the panoramas with.
    """
    self.bonus = bonus
    self.file_prefix = file_prefix
    self.files = [os.path.join(data_dir, '%s%03d.jpg' % (file_prefix, i + 1)) for i in range(num_images)]
    self.files = list(filter(os.path.exists, self.files))
    self.panoramas = None
    self.homographies = None
    print('found %d images' % len(self.files))

  def align_images(self, translation_only=False):
    """
    compute homographies between all images to a common coordinate system
    :param translation_only: see estimte_rigid_transform
    """
    # Extract feature point locations and descriptors.
    points_and_descriptors = []
    for file in self.files:
      image = sol4_utils.read_image(file, 1)
      self.h, self.w = image.shape
      pyramid, _ = sol4_utils.build_gaussian_pyramid(image, 3, 7)
      points_and_descriptors.append(find_features(pyramid))

    # Compute homographies between successive pairs of images.
    Hs = []
    for i in range(len(points_and_descriptors) - 1):
      points1, points2 = points_and_descriptors[i][0], points_and_descriptors[i + 1][0]
      desc1, desc2 = points_and_descriptors[i][1], points_and_descriptors[i + 1][1]

      # Find matching feature points.
      ind1, ind2 = match_features(desc1, desc2, .7)
      points1, points2 = points1[ind1, :], points2[ind2, :]

      # Compute homography using RANSAC.
      H12, inliers = ransac_homography(points1, points2, 100, 6, translation_only)

      # Uncomment for debugging: display inliers and outliers among matching points.
      # In the submitted code this function should be commented out!
      #display_matches(self.images[i], self.images[i+1], points1 , points2, inliers)

      Hs.append(H12)

    # Compute composite homographies from the central coordinate system.
    accumulated_homographies = accumulate_homographies(Hs, (len(Hs) - 1) // 2)
    self.homographies = np.stack(accumulated_homographies)
    self.frames_for_panoramas = filter_homographies_with_translation(self.homographies, minimum_right_translation=5)
    self.homographies = self.homographies[self.frames_for_panoramas]

  def generate_panoramic_images(self, number_of_panoramas):
    """
    combine slices from input images to panoramas.
    :param number_of_panoramas: how many different slices to take from each input image
    """
    if self.bonus:
      self.generate_panoramic_images_bonus(number_of_panoramas)
    else:
      self.generate_panoramic_images_normal(number_of_panoramas)

  def generate_panoramic_images_normal(self, number_of_panoramas):
    """
    combine slices from input images to panoramas.
    :param number_of_panoramas: how many different slices to take from each input image
    """
    assert self.homographies is not None

    # compute bounding boxes of all warped input images in the coordinate system of the middle image (as given by the homographies)
    self.bounding_boxes = np.zeros((self.frames_for_panoramas.size, 2, 2))
    for i in range(self.frames_for_panoramas.size):
      self.bounding_boxes[i] = compute_bounding_box(self.homographies[i], self.w, self.h)

    # change our reference coordinate system to the panoramas
    # all panoramas share the same coordinate system
    global_offset = np.min(self.bounding_boxes, axis=(0, 1))
    self.bounding_boxes -= global_offset

    slice_centers = np.linspace(0, self.w, number_of_panoramas + 2, endpoint=True, dtype=np.int)[1:-1]
    warped_slice_centers = np.zeros((number_of_panoramas, self.frames_for_panoramas.size))
    # every slice is a different panorama, it indicates the slices of the input images from which the panorama
    # will be concatenated
    for i in range(slice_centers.size):
      slice_center_2d = np.array([slice_centers[i], self.h // 2])[None, :]
      # homography warps the slice center to the coordinate system of the middle image
      warped_centers = [apply_homography(slice_center_2d, h) for h in self.homographies]
      # we are actually only interested in the x coordinate of each slice center in the panoramas' coordinate system
      warped_slice_centers[i] = np.array(warped_centers)[:, :, 0].squeeze() - global_offset[0]

    panorama_size = np.max(self.bounding_boxes, axis=(0, 1)).astype(np.int) + 1

    # boundary between input images in the panorama
    x_strip_boundary = ((warped_slice_centers[:, :-1] + warped_slice_centers[:, 1:]) / 2)
    x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
                                  x_strip_boundary,
                                  np.ones((number_of_panoramas, 1)) * panorama_size[0]])
    x_strip_boundary = x_strip_boundary.round().astype(np.int)

    self.panoramas = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0], 3), dtype=np.float64)
    for i, frame_index in enumerate(self.frames_for_panoramas):
      # warp every input image once, and populate all panoramas
      image = sol4_utils.read_image(self.files[frame_index], 2)
      warped_image = warp_image(image, self.homographies[i])
      x_offset, y_offset = self.bounding_boxes[i][0].astype(np.int)
      y_bottom = y_offset + warped_image.shape[0]

      for panorama_index in range(number_of_panoramas):
        # take strip of warped image and paste to current panorama
        boundaries = x_strip_boundary[panorama_index, i:i + 2]
        image_strip = warped_image[:, boundaries[0] - x_offset: boundaries[1] - x_offset]
        x_end = boundaries[0] + image_strip.shape[1]
        self.panoramas[panorama_index, y_offset:y_bottom, boundaries[0]:x_end] = image_strip

    # crop out areas not recorded from enough angles
    # assert will fail if there is overlap in field of view between the left most image and the right most image
    crop_left = int(self.bounding_boxes[0][1, 0])
    crop_right = int(self.bounding_boxes[-1][0, 0])
    assert crop_left < crop_right, 'for testing your code with a few images do not crop.'
    print(crop_left, crop_right)
    self.panoramas = self.panoramas[:, :, crop_left:crop_right, :]

  def generate_panoramic_images_bonus(self, number_of_panoramas):
    """
    The bonus
    :param number_of_panoramas: how many different slices to take from each input image
    """
    pass

  def save_panoramas_to_video(self):
    assert self.panoramas is not None
    out_folder = 'tmp_folder_for_panoramic_frames/%s' % self.file_prefix
    try:
      shutil.rmtree(out_folder)
    except:
      print('could not remove folder')
      pass
    os.makedirs(out_folder)
    # save individual panorama images to 'tmp_folder_for_panoramic_frames'
    for i, panorama in enumerate(self.panoramas):
      imwrite('%s/panorama%02d.png' % (out_folder, i + 1), panorama)
    if os.path.exists('%s.mp4' % self.file_prefix):
      os.remove('%s.mp4' % self.file_prefix)
    # write output video to current folder
    os.system('ffmpeg -framerate 3 -i %s/panorama%%02d.png %s.mp4' %
              (out_folder, self.file_prefix))


  def show_panorama(self, panorama_index, figsize=(20, 20)):
    assert self.panoramas is not None
    plt.figure(figsize=figsize)
    plt.imshow(self.panoramas[panorama_index].clip(0, 1))
    plt.show()

# sanity check:
def check_display_matches():
  images = []
  files = ['external\oxford1.jpg', 'external\oxford2.jpg']
  points_and_descriptors = []
  for file in files:
    image = sol4_utils.read_image(file, 1)
    images.append(image)
    h, w = image.shape
    pyramid, _ = sol4_utils.build_gaussian_pyramid(image, 3, 7)
    points_and_descriptors.append(find_features(pyramid))
  for i in range(len(points_and_descriptors) - 1):
    points1, points2 = points_and_descriptors[i][0], points_and_descriptors[i + 1][0]
    desc1, desc2 = points_and_descriptors[i][1], points_and_descriptors[i + 1][1]
    # Find matching feature points.
    ind1, ind2 = match_features(desc1, desc2, .7)
    points1, points2 = points1[ind1, :], points2[ind2, :]
    H12, inliers = ransac_homography(points1, points2, 100, 6, False)
    display_matches(images[0], images[1], points1 , points2, inliers)


# check_display_matches()
