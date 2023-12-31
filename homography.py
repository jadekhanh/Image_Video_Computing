# -*- coding: utf-8 -*-

!mkdir data

!mv *.jpg data/
!mv *.mat data/

"""1. **Optical Flow [10 pts, 3 parts]**

  **a) [2 pts]** Describe a scenario where the object is not moving but the optical flow field is not zero.

  **b) [3 pts]** The Constant Brightness Assumption (CBA) is used in the Lucas and Kanade Algorithm. Describe how the algorithm to handles the fact that the assumption might be violated.

  **c) [5 pts]** Why does the first order talor series provide a reasonable approximation for estimating optical flow?

Answers:

a) An instance where the object is not moving, but the optical flow is non-zero occurs when there is a change in the scene's perspective due to the camera's movement. An example scenario is person A stands still on the side of a road while person B walks towards them and holds a camera. As person B approaches person A, the size of person A in the camera's view changes, resulting in a non-zero optical flow, even though person A remains stationary.

b) The Lucas-Kanade algorithm handles CBA violations using the "Gaussian pyramid," creating a multi-scale image representation. The algorithm computes optical flow at each pyramid level and uses it as an initial estimate for the next level until the finest level. This algorithm continues until the flow is computed at the finest level of the pyramid. This approach detects and corrects errors at different scales.

c) The first-order Taylor series approximation is used to estimate optical flow because it reasonably approximates the relationship between the image intensity and its spatial and temporal derivatives. It bases on the assumption that the optical flow is small and that the brightness of a given point in an image remains constant over time. In addition, the Taylor series expansion, being a linear approximation, expresses the image intensity at a particular point as a function of its neighboring points, their spatial and temporal derivatives, and the optical flow between the two frames.

2. **Contour Detection [20 pts]**. In this problem we will build a basic contour detector.

  We have implemented a contour detector that uses the magnitude of the local image gradient as the boundary score as seen below:
"""

from PIL import Image
import numpy as np
import cv2, os
from scipy.io import loadmat
from scipy import signal
import matplotlib.pyplot as plt
import skimage
import evaluate_boundaries


N_THRESHOLDS = 99

def detect_edges(imlist, fn):
  """
  Detects edges in every image in the image list.

  :param imlist: a list of filenames.
  :param fn: an edge detection function.

  return: (list of image arrays, list of edge arrays). Edge array is of the same size as the image array.
  """
  images, edges = [], []
  for imname in imlist:
    I = cv2.imread(os.path.join('data', str(imname)+'.jpg'))
    images.append(I)

    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    I = I.astype(np.float32)/255.
    mag = fn(I)
    edges.append(mag)
  return images, edges

def evaluate(imlist, all_predictions):
  """
  Compares the predicted edges with the ground truth.

  :param imlist: a list of images.
  :param all_predictions: predicted edges for imlist.

  return: the evaluated F1 score.
  """
  count_r_overall = np.zeros((N_THRESHOLDS,))
  sum_r_overall = np.zeros((N_THRESHOLDS,))
  count_p_overall = np.zeros((N_THRESHOLDS,))
  sum_p_overall = np.zeros((N_THRESHOLDS,))
  for imname, predictions in zip(imlist, all_predictions):
    gt = loadmat(os.path.join('data', str(imname)+'.mat'))['groundTruth']
    num_gts = gt.shape[1]
    gt = [gt[0,i]['Boundaries'][0,0] for i in range(num_gts)]
    count_r, sum_r, count_p, sum_p, used_thresholds = \
              evaluate_boundaries.evaluate_boundaries_fast(predictions, gt,
                                                           thresholds=N_THRESHOLDS,
                                                           apply_thinning=True)
    count_r_overall += count_r
    sum_r_overall += sum_r
    count_p_overall += count_p
    sum_p_overall += sum_p

  rec_overall, prec_overall, f1_overall = evaluate_boundaries.compute_rec_prec_f1(
        count_r_overall, sum_r_overall, count_p_overall, sum_p_overall)

  return max(f1_overall)

def compute_edges_dxdy(I):
  """
  Returns the norm of dx and dy as the edge response function.

  :param I: image array

  return: edge array
  """

  dx = signal.convolve2d(I, np.array([[-1, 0, 1]]), mode='same')
  dy = signal.convolve2d(I, np.array([[-1, 0, 1]]).T, mode='same')
  mag = np.sqrt(dx**2 + dy**2)
  mag = normalize(mag)
  return mag

def normalize(mag):
  """
  Normalizes the edge array to [0,255] for display.

  :param mag: unnormalized edge array.

  return: normalized edge array
  """
  mag = mag / 1.5
  mag = mag * 255.
  mag = np.clip(mag, 0, 255)
  mag = mag.astype(np.uint8)
  return mag


imlist = [12084, 24077, 38092]
fn = compute_edges_dxdy
images, edges = detect_edges(imlist, fn)
display(Image.fromarray(np.hstack(images)))
display(Image.fromarray(np.hstack(edges)))
f1 = evaluate(imlist, edges)
print('Overall F1 score:', f1)

"""  **a) Warm-up [5 pts] .** As you visualize the produced edges, you will notice artifacts at image boundaries. Modify how the convolution is being done to minimize these artifacts."""

def compute_edges_dxdy_warmup(I):
  """
  Returns the norm of dx and dy as the edge response function.

  :param I: image array

  return: edge array, which is a HxW numpy array
  """

  """Hint: Look at arguments for scipy.signal.convolve2d"""
  # ADD YOUR CODE HERE
  dx = signal.convolve2d(I, np.array([[-1, 0, 1]]), mode='same', boundary='symm')
  dy = signal.convolve2d(I, np.array([[-1, 0, 1]]).T, mode='same', boundary='symm')

  mag = np.sqrt(dx**2 + dy**2)
  mag = normalize(mag)
  return mag

imlist = [12084, 24077, 38092]
fn = compute_edges_dxdy_warmup
images, edges = detect_edges(imlist, fn)
display(Image.fromarray(np.hstack(images)))
display(Image.fromarray(np.hstack(edges)))
f1 = evaluate(imlist, edges)
print('Overall F1 score:', f1)

""" **b) Smoothing [5 pts] .** Next, notice that we are using [−1, 0, 1] filters for computing the gradients, and they are susceptible to noise. Use derivative of Gaussian filters to obtain more robust estimates of the gradient. Experiment with different sigma for this Gaussian filtering and pick the one that works the best."""

def compute_edges_dxdy_smoothing(I):
  """
  Returns the norm of dx and dy as the edge response function.

  :param I: image array

  return: edge array, which is a HxW numpy array
  """

  """ Copy over your response from part a and alter it to include this answer. See cv2.GaussianBlur"""
  # ADD YOUR CODE HERE
  dx = signal.convolve2d(I, np.array([[-1, 0, 1]]), mode='same', boundary='symm')
  dy = signal.convolve2d(I, np.array([[-1, 0, 1]]).T, mode='same', boundary='symm')

  sigma = 100
  dx = cv2.GaussianBlur(dx, (7, 7), sigma)
  dy = cv2.GaussianBlur(dy, (7, 7), sigma)

  mag = np.sqrt(dx**2 + dy**2)
  mag = normalize(mag)
  return mag

imlist = [12084, 24077, 38092]
fn = compute_edges_dxdy_smoothing
images, edges = detect_edges(imlist, fn)
display(Image.fromarray(np.hstack(images)))
display(Image.fromarray(np.hstack(edges)))
f1 = evaluate(imlist, edges)
print('Overall F1 score:', f1)

"""**c) Non-maximum Suppression [10 pts] .** The current code does not produce thin edges. Implement non-maximum suppression, where we look at the gradient magnitude at the two neighbours in the direction perpendicular to the edge. We suppress the output at the current pixel if the output at the current pixel is not more than at the neighbors. You will have to compute the orientation of the contour (using the X and Y gradients), and then lookup values at the neighbouring pixels."""

def compute_edges_dxdy_nonmax(I):
  """
  Returns the norm of dx and dy as the edge response function after non-maximum suppression.

  :param I: image array

  return: edge array, which is a HxW numpy array
  """

  """ Copy over your response from part b and alter it to include this response"""
  # ADD YOUR CODE in part b
  dx = signal.convolve2d(I, np.array([[-1, 0, 1]]), mode='same', boundary='symm')
  dy = signal.convolve2d(I, np.array([[-1, 0, 1]]).T, mode='same', boundary='symm')

  sigma = 100
  dx = cv2.GaussianBlur(dx, (7, 7), sigma)
  dy = cv2.GaussianBlur(dy, (7, 7), sigma)

  # mag = numpy array that stores the gradient magnitudes of each pixel in the input image
  mag = np.sqrt(dx**2 + dy**2)
  mag = normalize(mag)

  # ADD YOUR CODE HERE
  angle = np.arctan2(dy, dx)  # angle = gradient orientation
  rows, cols = mag.shape      # Get the rows and cols of mag array

  # Loop over all pixels in the input image except for the pixels at the border
  for i in range(1, rows - 1):
      for j in range(1, cols - 1):
          # If current gradient orientation is close to horizontal (within -35 to 35 or -135 to 135)
          if (-np.pi/4 <= angle[i, j] < np.pi/4) or ((3/4)*np.pi <= angle[i, j] < (-3/4)*np.pi):
              # Pixels to the L & R are perpendicular to the edge
              # Use their magnitudes in non-maximum suppression step
              neighbor1, neighbor2 = mag[i, j-1], mag[i, j+1]
          else:
              # Pixels above and below are perpendicular to the edge
              # Use their magnitudes in non-maximum suppression step
              neighbor1, neighbor2 = mag[i-1, j], mag[i+1, j]

          # Check whether the current pixel is a candidate edge point
          if mag[i, j] <= neighbor1 or mag[i, j] <= neighbor2:
              mag[i, j] = 0

  non_max_mag = mag

  return non_max_mag

imlist = [12084, 24077, 38092]
fn = compute_edges_dxdy_nonmax
images, edges = detect_edges(imlist, fn)
display(Image.fromarray(np.hstack(images)))
display(Image.fromarray(np.hstack(edges)))
f1 = evaluate(imlist, edges)
print('Overall F1 score:', f1)

""" 3. **Stitching pairs of images [30 pts]**. In this problem we will estimate homography transforms to register and stitch image pairs. We are providing a image pairs that you should stitch together. We have also provided sample output, though, keep in mind that your output may look different from the reference output depending on implementation details.

   **Getting Started** We have implemented some helper functions to detect feature points in both images and extract descriptor of every keypoint in both images. We used SIFT descriptors from OpenCV library. You can refer to this [tutorial](https://docs.opencv.org/4.5.1/da/df5/tutorial_py_sift_intro.html) for more details about using SIFT in OpenCV. Please use opencv version 4.5 or later.
"""

# some helper functions

def imread(fname):
    """
    Read image into np array from file.

    :param fname: filename.

    return: image array.
    """
    return cv2.imread(fname)

def imread_bw(fname):
    """
    Read image as gray scale format.

    :param fname: filename.

    return: image array.
    """
    return cv2.cvtColor(imread(fname), cv2.COLOR_BGR2GRAY)

def imshow(img):
    """
    Show image.

    :param img: image array.
    """
    skimage.io.imshow(img)

def get_sift_data(img):
    """
    Detect the keypoints and compute their SIFT descriptors with opencv library

    :param img: image array.

    return: (keypoints array, descriptors). keypoints array (Nx2) contains the coordinate (x,y) of each keypoint.
    Descriptors have size Nx128.
    """
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    kp = np.array([k.pt for k in kp])
    return kp, des

def plot_inlier_matches(ax, img1, img2, inliers):
    """
    Plot the match between two image according to the matched keypoints

    :param ax: plot handle
    :param img1: left image
    :param img2: right image
    :param inliers: x,y in the first image and x,y in the second image (Nx4)
    """
    res = np.hstack([img1, img2])
    ax.set_aspect('equal')
    ax.imshow(res, cmap='gray')

    ax.plot(inliers[:,2], inliers[:,3], '+r')
    ax.plot(inliers[:,0] + img1.shape[1], inliers[:,1], '+r')
    ax.plot([inliers[:,2], inliers[:,0] + img1.shape[1]],
            [inliers[:,3], inliers[:,1]], 'r', linewidth=0.4)
    ax.axis('off')

"""**a) Putative Matches [5 pts].** Select putative matches based on the matrix of pairwise descriptor distances.  First, Compute distances between every descriptor in one image and every descriptor in the other image. We will use `scipy.spatial.distance.cdist(X, Y, 'sqeuclidean')`. Then, you can select all pairs  whose descriptor distances are below a specified threshold, or select the top few hundred descriptor pairs with the smallest pairwise distances. In your report, display the putative matches overlaid on the image pairs."""

import scipy

def get_best_matches(img1, img2, num_matches):
    """
    Returns the matched keypoints between img1 and img2.
    :param img1: left image.
    :param img2: right image.
    :param num_matches: the number of matches that we want.

    return: pixel coordinates of the matched keypoints (x,y in the second image and x,y in the first image), which is a Nx4 numpy array.
    """
    kp1, des1 = get_sift_data(img1)
    kp2, des2 = get_sift_data(img2)

    # Find distance between descriptors in images
    dist = scipy.spatial.distance.cdist(des1, des2, 'sqeuclidean')

    # Write your code to get the matches according to dist
    # ADD YOUR CODE HERE
    # 2 arrays of indices corresponding to the best matches between the descriptors in des1 and des2
    idx1, idx2 = np.unravel_index(np.argsort(dist, axis=None)[:num_matches], dist.shape)

    # Horizontally stack the keypoints in kp1 and kp2 corresponding to the best matches between 2 images
    out_kp = np.hstack((kp1[idx1], kp2[idx2]))

    return out_kp

img1 = imread('./data/left.jpg')
img2 = imread('./data/right.jpg')

data = get_best_matches(img1, img2, 300)
fig, ax = plt.subplots(figsize=(20,10))
plot_inlier_matches(ax, img1, img2, data)
#fig.savefig('sift_match.pdf', bbox_inches='tight')
print("Putative matches:", data)

"""**b) Homography Estimation and RANSAC [20 pts].** Implement RANSAC to estimate a homography mapping one image onto the other. Describe the various implementation details, and report all the hyperparameters, the number of inliers, and the average residual for the inliers (mean squared distance between the point coordinates in one image and the transformed coordinates of the matching point in the other image). Also, display the locations of inlier matches in both images.
        
  **Hints:** For RANSAC, a very simple implementation is sufficient. Use four matches to initialize the homography in each iteration. You should output a single transformation that gets the most inliers in the course of all the iterations. For the various RANSAC parameters (number of iterations, inlier threshold), play around with a few reasonable values and pick the ones that work best. Homography fitting calls for homogeneous least squares. The solution to the homogeneous least squares system $AX = 0$ is obtained from the SVD of $A$ by the singular vector corresponding to the smallest singular value. In Python, `U, S, V = numpy.linalg.svd(A)` performs the singular value decomposition (SVD). This function decomposes A into $U, S, V$ such that $A = USV$ (and not $USV^T$) and `V[−1,:]` gives the right singular vector corresponding to the smallest singular value. *Your implementation should not use any opencv functions.*

Description of RANSAC:

The function starts by initializing some variables, including the homography matrix H_out to None, a list best_perf to store the most number of inliers found, and a threshold value control the termination condition.

Then, the function iteratively runs RANSAC for 10000 iterations, randomly selecting four matches and producing a homography matrix h using the compute_homography function. For each iteration, the function calculates the residuals between the predicted pixel locations in the second image and the actual pixel locations, using the homography matrix h. If the residual is less than a threshold value of 1.5, then the corresponding match is considered an inlier and added to the inliers list.

The residual is calculated using several pixel coordinates. First, the pixel coordinates of the keypoint in the first image are obtained and stored in a matrix pixel_1, with an additional 1 to represent the homogeneous coordinate. Next, the predicted pixel in the second image is computed by multiplying the homography matrix h with pixel_1, and then normalizing it by dividing by the third component of the resulting vector, which ensures that the predicted pixel is represented in homogeneous coordinates. The resulting predicted pixel coordinates are stored in pred_pixel_2. Then, the pixel coordinates of the corresponding keypoint in the second image are obtained and stored in a matrix pixel_2. The residual between the two pixels is computed as the least square between pixel_2 and pred_pixel_2.

After all iterations, the function selects the homography matrix that produced the best performance, which has the most inliers and exceeds the minimum amount of inliners (=10). It then calculates the average residual of the inliers and returns the best homography matrix and the number of inliers found. Finally, the function prints the average residual and locations of the inliers.

Description of compute_homography:

The function first initializes an 8x9 matrix H and fills it with values computed from the matched keypoints. Each row of H corresponds to a single matching keypoint pair, and the columns represent the coefficients of the homography matrix.

The function then performs Singular Value Decomposition (SVD) on H to obtain the eigenvectors and eigenvalues of the matrix. The function returns the eigenvector corresponding to the smallest eigenvalue, which is the homography matrix h.

Finally, the reshape() function is used to convert the flattened 1D matrix v[8] into a 3x3 matrix to obtain the final homography matrix.
"""

from scipy.io._harwell_boeing import hb

def ransac(data, max_iters=10000, min_inliers=10):
    """
    Write your ransac code to find the best model, inliers, and residuals

    :param data: pixel coordinates of matched keypoints.
    :param max_iters: number of maximum iterations
    :param min_inliers: number of minimum inliers in RANSAC

    return: (homography matrix, number of inliers) -> (3x3 numpy array, int)
    """
    # ADD YOUR CODE HERE
    H_out = None
    best_perf = []
    threshold = 0.8
    total_residual = 0
    for index in range(max_iters):
      # get 4 random matches and produce homography h
      matches = data[np.random.choice(len(data), size=4, replace=False)]
      inliers = []
      h = compute_homography(matches)

      for i in range(len(data)):
        # find the residuals
        pixel_1 = np.transpose(np.matrix([data[i][0], data[i][1], 1]))                      # pixel coordinates of the keypoint in img1
        pred_pixel_2 = np.dot(h, pixel_1) / np.dot(h, pixel_1)[2]                           # the predicted pixel in img2 using h & normalize it
        pixel_2 = np.transpose(np.matrix([data[i][2], data[i][3], 1]))                      # pixel coordinates of the keypoint in img2
        residual = abs(np.linalg.norm(pixel_2 - pred_pixel_2))                              # residual between pixel and predicted pixel
        if residual <= threshold:
            inliers.append(data[i])
            total_residual += residual**4
      # find the best peformance model, H, and average residual of inliers
      if len(best_perf) < len(inliers):
        if min_inliers < len(inliers):
          H_out = h
          best_perf = inliers
          avg_residual = total_residual / len(inliers)
          num_inliers = len(best_perf)
      # break when the number of inliers reaches 80% of data
      if threshold * len(data) < len(best_perf):
        break

    print("Average residual:", avg_residual)
    print("Locations of Inliers", best_perf)
    return H_out, num_inliers

def compute_homography(matches):
    """
    Write your code to compute homography according to the matches

    :param matches: coordinates of matched  keypoints.

    return: homography matrix, which is a 3x3 numpy array
    """
    # ADD YOUR CODE HERE
    H = np.zeros((8,9))
    for i, match in enumerate(matches):
      x1, y1, x2, y2 = match

      H[2 * i, :] = [-x1, -y1, -1, 0, 0, 0, x2 * x1, x2 * y1, x2]
      H[2 * i + 1, :] = [0, 0, 0, -x1, -y1, -1, y2 * x1, y2 * y1, y2]

    u,s,v = np.linalg.svd(H)
    h = np.reshape(v[8], (3, 3))    # reshape into 3x3
    return h

np.random.seed(1237)
# Report the number of inliers in your matching
H, max_inliers = ransac(data)
print("Inliers:", max_inliers)

"""**c) Image Warping [5 pts].** Warp one image onto the other using the estimated transformation. You can use opencv functions for this purpose."""

def warp_images(img1, img2, H):
    """
    Write your code to stitch images together according to the homography

    :param img1: left image
    :param img2: right image
    :param H: homography matrix

    return: stitched image, should be a HxWx3 array.
    """
    # ADD YOUR CODE HERE
    h1, w1, c1 = img1.shape
    h2, w2, c2 = img2.shape

    # Declare the dimension of img_warped
    h = h1
    w = w1 + w2

    # Apply inverse homography transformation to img2 and create img_warped
    inverse_H = np.linalg.inv(H)
    img_warped = cv2.warpPerspective(img2, inverse_H, (w, h))
    img_warped[0:h1, 0:w1] = img1

    return img_warped

# display the stitching results
img_warped = warp_images(img1, img2, H)
display(Image.fromarray(img_warped))

"""**d) Image Warping Bonus [5 bonus pts].** Warp one image onto the other using the estimated transformation *without opencv functions*. Create a new image big enough to hold the panorama and composite the two images into it. You can composite by averaging the pixel values where the two images overlap, or by using the pixel values from one of the images. Your result should look similar to the sample output. You should create **color panoramas** by applying  the same compositing step to each of the color channels separately (for estimating the transformation, it is sufficient to use grayscale images). You may find `ProjectiveTransform` and warp functions in `skimage.transform` useful."""

import skimage.transform


def warp_images_noopencv(img1, img2, H):
    """
    Write your code to stitch images together according to the homography

    :param img1: left image
    :param img2: right image
    :param H: homography matrix

    return: stitched image, should be a HxWx3 array.
    """
    # ADD YOUR CODE HERE
    h1, w1, c1 = img1.shape
    h2, w2, c2 = img2.shape

    # Declare the dimension of img_warped
    h = h1
    w = w1 + w2
    c = c1

    # Apply projective transformation to img2
    transform = skimage.transform.ProjectiveTransform(H)
    trans_img2 = skimage.transform.warp(image=img2, inverse_map=transform, map_args={}, output_shape=(h, w, c), order=None, mode='constant', cval=0.0, clip=True, preserve_range=False) * 255

    # Create a new image by combining img1 and img2_trans
    trans_img1 = np.zeros((h, w, c))
    trans_img1[:, :w1] = img1                                             # assign trans_img1 to the left part of the trans_img1
    img_warped_nocv = np.where(trans_img2 != 0, trans_img2, trans_img1)   # take img2_trans where it is available and tke img1_trans where img2_trans isn't available

    return img_warped_nocv

# display and report the stitching results
img_warped_nocv = warp_images_noopencv(img1, img2, H)
display(Image.fromarray(img_warped))
