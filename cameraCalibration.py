# -*- coding: utf-8 -*-

"""
1. **Camera Calibration [8 pts]**. For the pair of images in the folder `calibraion`, calculate the camera projection matrices by using 2D matches in both views and 3D point
coordinates in `lab_3d.txt`. Once you have computed your projection matrices,
you can evaluate them using the provided evaluation function
(evaluate_points). The function outputs the projected 2-D points and
residual error. Report the estimated 3 × 4 camera projection matrices (for
each image), and residual error.
<b>Hint:</b> The residual error should be < 20 and the squared distance of the
projected 2D points from actual 2D points should be < 4.

A.


B. From Lecture 21, we know the expression of 2 epipoles are: \\

First epipole:

 $F e^{\prime}=0 and F e=0 \quad F=[\mathbf{a}]_{\times} A, F^T=A^T[\mathbf{a}]_{\times}^T $ \\

Since $[\mathbf{a}]_{\times}$is skew-symmetric, $[\mathbf{a}]_{\times}^T=-[\mathbf{a}]_{\times} ,\\
F^T a=A^T[\mathbf{a}]_{\times}^T a=-A^T[\mathbf{a}]_{\times} a=-A \mathbf{0}$ \\

Because $[\mathbf{a}]_{\times} a=\mathbf{0}$, a is the right epipole in the figure below:
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def evaluate_points(M, points_2d, points_3d):
    """
    Visualize the actual 2D points and the projected 2D points calculated from
    the projection matrix
    You do not need to modify anything in this function, although you can if you
    want to
    :param M: projection matrix 3 x 4
    :param points_2d: 2D points N x 2
    :param points_3d: 3D points N x 3
    :return:
    """
    N = len(points_3d)
    points_3d = np.hstack((points_3d, np.ones((N, 1))))
    points_3d_proj = np.dot(M, points_3d.T).T
    u = points_3d_proj[:, 0] / points_3d_proj[:, 2]
    v = points_3d_proj[:, 1] / points_3d_proj[:, 2]
    residual = np.sum(np.hypot(u-points_2d[:, 0], v-points_2d[:, 1]))
    points_3d_proj = np.hstack((u[:, np.newaxis], v[:, np.newaxis]))
    return points_3d_proj, residual

# Write your code here for camera calibration
def camera_calibration(pts_2d, pts_3d):
    """
    write your code to compute camera matrix
    """
    # <YOUR CODE>
    A = []
    for i in range(len(pts_3d)):
        # 3D coordinates of the ith point in pts_3d
        x3 = pts_3d[i, 0]
        y3 = pts_3d[i, 1]
        z3 = pts_3d[i, 2]

        # 2D coordinates of ith point in pts_3d
        x2 = pts_2d[i, 0]
        y2 = pts_2d[i, 1]

        # Create matrix A
        A.append( [x3, y3, z3, 1, 0, 0, 0, 0, -x2*x3, -x2*y3, -x2*z3, -x2] )
        A.append( [0, 0, 0, 0, x3, y3, z3, 1, -y2*x3, -y2*y3, -y2*z3, -y2] )

    A = np.asarray(A)

    u, s, v = np.linalg.svd(A)
    H = v[-1, :].reshape(3, 4)
    return H

# Load 3D points, and their corresponding locations in
# the two images.
pts_3d = np.loadtxt('lab_3d.txt')
matches = np.loadtxt('lab_matches.txt')

# print lab camera projection matrices:
lab1_proj = camera_calibration(matches[:, :2], pts_3d)
lab2_proj = camera_calibration(matches[:, 2:], pts_3d)
print('lab 1 camera projection')
print(lab1_proj)

print('')
print('lab 2 camera projection')
print(lab2_proj)

# evaluate the residuals for both estimated cameras
_, lab1_res = evaluate_points(lab1_proj, matches[:, :2], pts_3d)
print('residuals between the observed 2D points and the projected 3D points:')
print('residual in lab1:', lab1_res)
_, lab2_res = evaluate_points(lab2_proj, matches[:, 2:], pts_3d)
print('residual in lab2:', lab2_res)

"""2. **Camera Centers [3 pts].** Calculate the camera centers using the
estimated or provided projection matrices. Report the 3D
locations of both cameras in your report. <b>Hint:</b> Recall that the
camera center is given by the null space of the camera matrix.

We know that the two epipoles of any stereo system can be expressed as:

$$
\begin{array}{lr}
F e^{\prime}=0, & F e=0 \\
F=[\mathbf{a}]_{\times} A, & F^{T}=A^{T}[\mathbf{a}]_{\times}^{T}
\end{array}
$$

Because $[\mathbf{a}]_{\times}$is skew-symmetric, $[\mathbf{a}]_{\times}^{T}=-[\mathbf{a}]_{\times}$Thus, we can simply plug in a in for $e$ and $e^{\prime}$ :

$$
\begin{aligned}
F^{T} a & =A^{T}[\mathbf{a}]_{\times}^{T} a \\
& =-A^{T}[\mathbf{a}]_{\times} a \\
& =-A \mathbf{0}
\end{aligned}
$$

Because $[\mathbf{a}]_{\times} a=\mathbf{0}$, a must clearly be the right epipole in figure 3 .
"""

import scipy.linalg
# Write your code here for computing camera centers
def calc_camera_center(proj):
    """
    write your code to get camera center in the world
    from the projection matrix
    """
    # <YOUR CODE>
    A = proj[:3, :3]
    R, Q = scipy.linalg.rq(A)
    R_norm = R / R[2,2]

    for i in range(2):
        if R_norm[i, i] < 0:
            R_norm[:, i] *= -1
            Q[i, :] *= -1

    M = R_norm @ Q
    T = np.linalg.inv(R_norm) @ (M[0, 0] * proj / proj[0, 0])[:, 3]
    C = -Q.T @ T
    return C

# compute the camera centers using
# the projection matrices
lab1_c = calc_camera_center(lab1_proj)
lab2_c = calc_camera_center(lab2_proj)
print('lab1 camera center', lab1_c)
print('lab2 camera center', lab2_c)

"""3. **Triangulation [8 pts].** Use linear least squares to triangulate the
3D position of each matching pair of 2D points using the two camera
projection matrices. As a sanity check, your triangulated 3D points for the
lab pair should match very closely the originally provided 3D points in
`lab_3d.txt`. Display the two camera centers and
reconstructed points in 3D. Include snapshots of this visualization in your
report. Also report the residuals between the observed 2D points and the
projected 3D points in the two images. Note: You do not
need the camera centers to solve the triangulation problem. They are used
just for the visualization.


"""

# Write your code here for triangulation
from mpl_toolkits.mplot3d import Axes3D

def triangulation(lab_pt1, lab1_proj, lab_pt2, lab2_proj):
    """
    write your code to triangulate the points in 3D
    """
    triangulated = np.zeros((lab_pt1.shape[0],3))
    A = np.zeros((4,4))
    for i in range(lab_pt1.shape[0]):
      x1, y1 = lab_pt1[i]
      x2, y2 = lab_pt2[i]
      A[:2, :] = np.array([[1, 0, -x1], [0, 1, -y1]]) @ lab1_proj
      A[2:, :] = np.array([[1, 0, -x2], [0, 1, -y2]]) @ lab2_proj
      u, s, v = np.linalg.svd(A)
      X = v[-1, :]
      X /= X[-1]
      triangulated[i, :] = X[:3]
    return triangulated

def evaluate_points_3d(points_3d_lab, points_3d_gt):
    """
    write your code to evaluate the triangulated 3D points
    """
    diff = points_3d_lab - points_3d_gt
    squared_error = (diff ** 2).sum(axis=1)
    return squared_error

# triangulate the 3D point cloud for the lab data
matches_lab = np.loadtxt('lab_matches.txt')
lab_pt1 = matches_lab[:,:2]
lab_pt2 = matches_lab[:,2:]
points_3d_gt = np.loadtxt('lab_3d.txt')
points_3d_lab = triangulation(lab_pt1, lab1_proj, lab_pt2, lab2_proj)
res_3d_lab = evaluate_points_3d(points_3d_lab, points_3d_gt)
print('Mean 3D reconstuction error for the lab data: ', round(np.mean(res_3d_lab), 5))
_, res_2d_lab1 = evaluate_points(lab1_proj, lab_pt1, points_3d_lab)
_, res_2d_lab2 = evaluate_points(lab2_proj, lab_pt2, points_3d_lab)
print('2D reprojection error for the lab 1 data: ', np.mean(res_2d_lab1))
print('2D reprojection error for the lab 2 data: ', np.mean(res_2d_lab2))
# visualization of lab point cloud
camera_centers = np.vstack((lab1_c, lab2_c))
print(camera_centers.shape)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points_3d_lab[:, 0], points_3d_lab[:, 1], points_3d_lab[:, 2], c='b', label='Points')
ax.scatter(camera_centers[:, 0], camera_centers[:, 1], camera_centers[:, 2], c='g', s=50, marker='^', label='Camera Centers')
ax.legend(loc='best')

"""4. **Extra Credits [3 pts].** Use the putative match generation and RANSAC
code from `PS3` to estimate fundamental matrices without
ground-truth matches. For this part, only use the normalized algorithm.
Report the number of inliers and the average residual for the inliers.
Compare the quality of the result with the one you get from ground-truth
matches.


"""



"""5. **Epipolar Geometry [15 pts total].** Let $M1$ and $M2$ be two camera matrices. We know that the fundamental matrix corresponding to these camera matrices is of the following form:
$$F = [a]×A,$$
where $[a]×$ is the matrix
$$[a]× = \begin{bmatrix}
0 & ay & −az
−ay & 0 & ax
az & −ax & 0\end{bmatrix}.$$
Assume that $M1 = [I|0]$ and $M2 = [A|a]$, where $A$ is a 3 × 3 (nonsingular) matrix.

  1. **Combining with optical flow [5 pts]**. Propose an approach to modifies your optical flow implementation from Lab 8 to use epipolar geometry.

  2. **Epipoles [10 pts]** Prove that the last column of $M2$, denoted by $a$, is one of the epipoles and draw your result in a diagram similar to the following image:

Answers for Question 5:

1. First, we compute the fundamental matrix F using the formula above. For each pixel in the 1st image, we find its corresponding epipolar line in the 2nd image using the formula l' = Fx, where x is the location in homogeneous coordinates. Next, we search along the epipolar line in the 2nd image to find the pixel with the best match to the pixel in the 1st image using Lucas-Kenade algorithm. Then, we compute the disparity between the matches pixels in the two images. We repeat the steps above for all pixels in the first image to obtain the disparity map.

2.
"""

from PIL import Image
from IPython.display import display
display(Image.open('epipolar.jpg'))

"""6. **3D Estimation [Extra credit - 10 pts bonus].** Design a bundle adjuster that allows for arbitrary chains of transformations and prior knowledge about the unknowns, see [SZ Figures 11.14-11.15](http://szeliski.org/Book/) for an example.

7. **Vanishing points [12 pts total]** Using `ps5_example.jpg`, you need to estimate the three major orthogonal vanishing points. Use at least three manually selected lines to solve for each vanishing point. The starter code below provides an interface for selecting and drawing the lines, but the code for computing the vanishing point needs to be inserted. For details on estimating vanishing points, see Lab 10.
"""

# %matplotlib tk
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def get_input_lines(im, min_lines=3):
    """
    Allows user to input line segments; computes centers and directions.
    Inputs:
        im: np.ndarray of shape (height, width, 3)
        min_lines: minimum number of lines required
    Returns:
        n: number of lines from input
        lines: np.ndarray of shape (3, n)
            where each column denotes the parameters of the line equation
        centers: np.ndarray of shape (3, n)
            where each column denotes the homogeneous coordinates of the centers
    """
    n = 0
    lines = np.zeros((3, 0))
    centers = np.zeros((3, 0))

    plt.figure()
    plt.axis('off')
    plt.imshow(im)
    print(f'Set at least {min_lines} lines to compute vanishing point')
    print(f'The delete and backspace keys act like right clicking')
    print(f'The enter key acts like middle clicking')
    while True:
        print('Click the two endpoints, use the right button (delete and backspace keys) to undo, and use the middle button to stop input')
        clicked = plt.ginput(2, timeout=0, show_clicks=True)
        if not clicked or len(clicked) < 2:
            if n < min_lines:
                print(f'Need at least {min_lines} lines, you have {n} now')
                continue
            else:
                # Stop getting lines if number of lines is enough
                break

        # Unpack user inputs and save as homogeneous coordinates
        pt1 = np.array([clicked[0][0], clicked[0][1], 1])
        pt2 = np.array([clicked[1][0], clicked[1][1], 1])
        # Get line equation using cross product
        # Line equation: line[0] * x + line[1] * y + line[2] = 0
        line = np.cross(pt1, pt2)
        lines = np.append(lines, line.reshape((3, 1)), axis=1)
        # Get center coordinate of the line segment
        center = (pt1 + pt2) / 2
        centers = np.append(centers, center.reshape((3, 1)), axis=1)

        # Plot line segment
        plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color='b')

        n += 1

    return n, lines, centers

def plot_lines_and_vp(ax, im, lines, vp):
    """
    Plots user-input lines and the calculated vanishing point.
    Inputs:
        im: np.ndarray of shape (height, width, 3)
        lines: np.ndarray of shape (3, n)
            where each column denotes the parameters of the line equation
        vp: np.ndarray of shape (3, )
    """
    bx1 = min(1, vp[0] / vp[2]) - 10
    bx2 = max(im.shape[1], vp[0] / vp[2]) + 10
    by1 = min(1, vp[1] / vp[2]) - 10
    by2 = max(im.shape[0], vp[1] / vp[2]) + 10

    ax.imshow(im)
    for i in range(lines.shape[1]):
        if lines[0, i] < lines[1, i]:
            pt1 = np.cross(np.array([1, 0, -bx1]), lines[:, i])
            pt2 = np.cross(np.array([1, 0, -bx2]), lines[:, i])
        else:
            pt1 = np.cross(np.array([0, 1, -by1]), lines[:, i])
            pt2 = np.cross(np.array([0, 1, -by2]), lines[:, i])
        pt1 = pt1 / pt1[2]
        pt2 = pt2 / pt2[2]
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'g')

    ax.plot(vp[0] / vp[2], vp[1] / vp[2], 'ro')
    ax.set_xlim([bx1, bx2])
    ax.set_ylim([by2, by1])

def get_top_and_bottom_coordinates(im, obj):
    """
    For a specific object, prompts user to record the top coordinate and the bottom coordinate in the image.
    Inputs:
        im: np.ndarray of shape (height, width, 3)
        obj: string, object name
    Returns:
        coord: np.ndarray of shape (3, 2)
            where coord[:, 0] is the homogeneous coordinate of the top of the object and coord[:, 1] is the homogeneous
            coordinate of the bottom
    """
    plt.figure()
    plt.imshow(im)

    print('Click on the top coordinate of %s' % obj)
    clicked = plt.ginput(1, timeout=0, show_clicks=True)
    x1, y1 = clicked[0]
    # Uncomment this line to enable a vertical line to help align the two coordinates
    plt.plot([x1, x1], [0, im.shape[0]], 'b')
    print('Click on the bottom coordinate of %s' % obj)
    clicked = plt.ginput(1, timeout=0, show_clicks=True)
    x2, y2 = clicked[0]

    plt.plot([x1, x2], [y1, y2], 'b')

    return np.array([[x1, x2], [y1, y2], [1, 1]])

"""7.1. **Estimating Horizon [3 pts]** You should: a) plot the VPs and the lines used to estimate the vanishing points (VP) on the image plane using the provided code. b) Specify the VP pixel coordinates. c) Plot the ground horizon line and specify its parameters in the form $a * x + b * y + c = 0$. Normalize the parameters so that: $a^2 + b^2 = 1$."""

def get_vanishing_point(lines):
    """
    Solves for the vanishing point using the user-input lines.
    """
    # get intersection of input lines
    intersect1 = np.cross(lines[:, 0], lines[:, 1])
    intersect2 = np.cross(lines[:, 1], lines[:, 2])
    intersect3 = np.cross(lines[:, 0], lines[:, 2])
    # convert to homogeneous coordinate
    intersect1 /= intersect1[-1]
    intersect2 /= intersect2[-1]
    intersect3 /= intersect3[-1]

    intersections = np.vstack((intersect1, intersect2, intersect3))
    vp = np.mean(intersections, axis=0)
    print('vanishing point:', vp)
    return vp

def get_horizon_line(vpts):
    """
    Calculates the ground horizon line.
    """
    vx, vy = vpts[:, 0], vpts[:, 1]
    a, b, c = np.cross(vx, vy)
    scale = np.sqrt(1 / (a**2 + b**2))
    line = np.array([a, b, c]) * scale

    print('Horizon Line:', line)
    return line

def plot_horizon_line(img, vpts):
    """
    Plots the horizon line.
    """
    plt.figure()
    plt.imshow(img)
    pt1, pt2 = vpts[:, 0], vpts[:, 1]
    plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'g')
    plt.plot(pt1[0], pt1[1], 'ro')
    plt.plot(pt2[0], pt2[1], 'ro')
    plt.title('Horizon Line')
    plt.savefig('horizon_line.jpg')
    plt.show()
    return

im = np.asarray(Image.open('ps5_example.jpg'))

# Get vanishing points for each of the directions
num_vpts = 3
vpts = np.zeros((3, num_vpts))
for i in range(num_vpts):
    print('Getting vanishing point %d' % i)
    # Get at least three lines from user input
    n, lines, centers = get_input_lines(im)
    # <YOUR IMPLEMENTATION> Solve for vanishing point
    vpts[:, i] = get_vanishing_point(lines)
    # Plot the lines and the vanishing point
    plot_lines_and_vp(im, lines, vpts[:, i], i)

# <YOUR IMPLEMENTATION> Get the ground horizon line
horizon_line = get_horizon_line(vpts)
# <YOUR IMPLEMENTATION> Plot the ground horizon line
plot_horizon_line(im, vpts)

"""7.2. **Solving for camera parameters [3 pts]** Using the fact that the vanishing directions are orthogonal, solve for the focal length and optical center (principal point) of the camera. Show all your work and include the computed parameters in your report."""

from sympy import Symbol, Matrix, solve
def get_camera_parameters(vpts):
    """
    Computes the camera parameters. Hint: The SymPy package is suitable for this.
    """
    pt1, pt2, pt3 = Matrix(vpts[:, 0]), Matrix(vpts[:, 1]), Matrix(vpts[:, 2])
    f, px, py = Symbol('f'), Symbol('px'), Symbol('py')
    K = Matrix( ((f, 0, px), (0, f, py), (0, 0, 1)) )
    K_inv = K.inv()
    Eq1 = pt1.T * K_inv.T * K_inv * pt2
    Eq2 = pt1.T * K_inv.T * K_inv * pt3
    Eq3 = pt2.T * K_inv.T * K_inv * pt3
    res = solve([Eq1, Eq2, Eq3], [f, px, py])
    print("Camera parameters", res[0])

    f, px, py = res[0]
    return f, px, py

# <YOUR IMPLEMENTATION> Solve for the camera parameters (f, u, v)
f, u, v = get_camera_parameters(vpts)

"""7.3. **Camera rotation matrix [3 pts]** Compute the rotation matrix for the camera, setting the vertical vanishing point as the Y-direction, the right-most vanishing point as the X-direction, and the left-most vanishing point as the Z-direction."""

def get_rotation_matrix(vpts, f, px, py):
    """
    Computes the rotation matrix using the camera parameters.
    """
    vpx, vpy, vpz = vpts[:, 1], vpts[:, 2], vpts[:, 0]
    K = np.array([[f, 0, px], [0, f, py], [0, 0, 1]]).astype(np.float)
    K_inv = np.linalg.inv(K)

    r1 = np.matmul(K_inv, vpx)
    r2 = np.matmul(K_inv, vpy)
    r3 = np.matmul(K_inv, vpz)

    R1 = 1 / np.linalg.norm(r1) * r1
    R2 = 1 / np.linalg.norm(r2) * r2
    R3 = 1 / np.linalg.norm(r3) * r3

    R = np.vstack((R1, R2, R3)).T
    print('Rotation Matrix:', R)
    return R

# <YOUR IMPLEMENTATION> Solve for the rotation matrix
R = get_rotation_matrix(vpts, f, px, py)

"""7.4. **Measurement estimation [3 pts]** Estimate the heights of (a) the large building in the center of the image, (b) the spike statue, and (c) the lamp posts assuming that the person nearest to the spike is 5ft 6in tall. In the report, show all the lines and measurements used to perform the calculation. How do the answers change if you assume the person is 6ft tall?"""

def estimate_height(coord_ref, coord_obj, vpts, H_ref):
    """
    Estimates height for a specific object using the recorded coordinates. You might need to plot additional images here for
    your report.
    """
    vpx, vpy, vpz = vpts[:, 0], vpts[:, 1], vpts[:, 2]
    r, b = coord_obj[:, 0], coord_obj[:, 1]
    t0, b0 = coord_ref[:, 0], coord_ref[:, 1]
    v = np.cross(np.cross(b, b0), np.cross(vpx, vpy))
    t = np.cross(np.cross(v, t0), np.cross(r, b))
    t /= t[-1]

    ratio = np.linalg.norm(t-b) * np.linalg.norm(vpz-r) / (np.linalg.norm(r-b) * np.linalg.norm(vpz-t))
    print('Image cross ratio =', '{:.2f}'.format(ratio))
    height = H_ref / ratio
    print('Height =', '{:.2f}'.format(height), 'm')
    return height

# Record image coordinates for each object and store in map
objects = ('person', 'Building', 'the spike statue', 'the lamp posts')
coords = dict()
for obj in objects:
    coords[obj] = get_top_and_bottom_coordinates(im, obj)

# <YOUR IMPLEMENTATION> Estimate heights
for obj in objects[1:]:
    print('Estimating height of %s' % obj)
    height = estimate_height(coord_ref, coord_obj, vpts, H_ref)

"""8. **Warped view [5 bonus pts]** Compute and display rectified views of the ground plane and the large building in the center of the image."""
