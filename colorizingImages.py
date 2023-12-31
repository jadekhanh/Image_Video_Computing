# -*- coding: utf-8 -*-
"""
**Colorizing images [30 points total, 3 parts].**

  In this problem, we will learn to work with images by taking the digitized [Prokudin-Gorskii glass plate images](https://www.loc.gov/exhibits/empire/gorskii.html) and automatically producing a color image with as few visual artifacts as possible. In order to do this, you will need to extract the three color channel images, place them on top of each other, and align them so that they form a single RGB color image.

  **a)** &ensp; **Read images [5 pts].** We'll start simple. Our first task is to read the file [00351v.jpg](https://drive.google.com/file/d/11fwxjlZkDOApoVZx0Pr4am1ClA6qdNaY/view?usp=sharing), extract the three color channel images and display each of them. Note that the filter order from top to bottom is BGR.
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def crop_bgr_from_img(img_path: str)->[np.array,np.array,np.array]:
    """
     Read the file as a numpy array and crop the three color channel images.
     Note that the filter order from top to bottom is BGR.

     :param img_path: the path to the image file, which is a string.

     :return: blue, green and red color channel images. Each of them is a 341x396 numpy array.
    """

    img=np.array(Image.open(img_path),dtype=np.uint16)

    # Renormlize the pixel value to be in range [0,255].
    img=255*(img*1.0-np.min(img))/(np.max(img)-np.min(img))
    img=np.uint8(img)

    h,w=img.shape

    # Extract the three color channel images from top to bottom.
    seg_h=h//3

    # ADD YOUR CODE HERE (5 pts)
    # slice individual images into thirds vertically
    top_third = img[:seg_h,:]
    middle_third = img[seg_h:2*seg_h,:]
    bottom_third = img[2*seg_h:2*seg_h + seg_h, :]

    blue = top_third
    green = middle_third
    red = bottom_third

    # Make sure the three color channel images have the same shape
    assert blue.shape==green.shape==red.shape

    return blue,green,red

blue,green,red=crop_bgr_from_img('00351v.jpg')

# Display three color channel images
plt.figure(figsize=(20,20))
for i,img_arr in enumerate([blue,green,red]):
    img=Image.fromarray(img_arr)
    plt.subplot(1,3,i+1)
    plt.axis('off')
    plt.imshow(img,cmap='gray')

"""b)** &ensp; **Basic alignment [10 pts].** Now that we've divided the image into three channels, the next thing to do is to align two of the channels to the third. The easiest way to align the parts is to exhaustively search over a window of possible displacements (say [-15,15] pixels independently for the x and y axis), score each one using some image matching metric, and take the displacement with the best score. We can use normalized cross-correlation (NCC) as the image matching metric, which is simply the dot product between the two images normalized to have zero mean and unit norm."""

def ncc(img_a: np.array,img_b: np.array)->float:
    """
     Compute the normalized cross-correlation between two color channel images
     and return the matching score.

     :param img_a: the first image, which is a 341x396 numpy array.
     :param img_b: the second image, which is a 341x396 numpy array.

     :return: the normalized cross-correlation score.
    """

    ncc=0

    # ADD YOUR CODE HERE (5 pts)
    mean1 = np.mean(img_a)
    std1 = np.std(img_a)
    mean2 = np.mean(img_b)
    std2 = np.std(img_b)
    ncc = np.sum((img_a- mean1)*(img_b - mean2)/(std1*std2))

    return ncc

"""Then, we align two color channel images by exhaustively searching over a window of possible displacements, score each one using NCC, and take the displacement with the best score. (Hint: you can use np.roll function to shift the entire image by the specified displacement."""

def align_imga_to_imgb(img_a:np.array,img_b:np.array, wd_size:int=15)->(np.array,(int,int)):
    """
     Align two color channel images. Return the aligned image_a and its displacement.

     :param img_a: the image to be shifted, which is a 341x396 numpy array.
     :param img_b: the image that is fixed, which is a 341x396 numpy array.

     :return: a tuple of (aligned_a, displacement_of_a). ''aligned_a'' is the aligned image_a,
     which is a 341x396 numpy array.''displacement_of_a'' is the displacement vector of img_a, which is
     a tuple of (row displacement, column displacement).
    """

    # Initialize the image matching score.
    score=0

    # Initialize the aligned image A.
    aligned_a=None

    #Initialize the displacement vector.
    displacement=(0,0)

    # Shift image A whithin range [-wd_size, wd_size], score each shifted image
    # and take the one with the best score.
    for i in range(-wd_size,wd_size):
        for j in range(-wd_size,wd_size):
            # Shift img_a's rows by i pixels, columns by j pixels
            # ADD YOUR CODE HERE (5 pts)
            # axis = 0 because of the rows, axis = 1 because of the columns
            shifted_a = np.roll(np.roll(img_a, i, axis = 0), j, axis = 1)

            new_score=ncc(shifted_a,img_b)

            if new_score>score:
                score=new_score
                aligned_a=shifted_a
                displacement=[i,j]
    return  aligned_a,displacement

"""Finally, we can display the colorized output and the (x,y) displacement vector that were used to align the channels."""

def colorize_image(b:np.array,g:np.array,r:np.array)->(np.array,list):
    """
     Align the three color channel images. Return the colored image
     and a list of the displacement vector for each channel.

     :param b: the blue channel image, which is a 341x396 numpy array.
     :param g: the greeb channel image, which is a 341x396 numpy array.
     :param r: the red channel image, which is a 341x396 numpy array.

     :return: a tuple of (colored_image, displacements). ''colored_image'' is a 341x396x3 numpy array.
     ''displacements'' is a list of the displacement vector for each channel.
    """

    # Align the red and blue channels to the green channel.
    aligned_r,dis_r = align_imga_to_imgb(r,g)
    aligned_b,dis_b = align_imga_to_imgb(b,g)

    aligned_g,dis_g =g,(0,0)

    # Combine the aligned channels to a color image.
    colored_img=np.stack([aligned_r,aligned_g,aligned_b],axis=2)

    return colored_img,[dis_r,dis_g,dis_b]

colored_img,displacements=colorize_image(blue,green,red)
print('Displacement of aligning the red channel to the green channel:',displacements[0])
print('Displacement of aligning the green channel to the green channel:',displacements[1])
print('Displacement of aligning the blue channel to the green channel:',displacements[2])
plt.figure(figsize=(5,5))
plt.axis('off')
plt.imshow(colored_img)

"""**c)** &ensp; **Multiscale alignment [15 pts].** Now let's try colorizing the high-resolution image [01047u.tif](https://drive.google.com/file/d/1HbmTOLAw_f64wurxJyorOraKGX6EIree/view?usp=sharing). This image is of size 9656 x 3741. Therefore, exhaustive search over all possible displacements will become prohibitively expensive. To deal with this case, we can implement a faster search procedure using an image pyramid. An image pyramid represents the image at multiple scales (usually scaled by a factor of 2) and the processing is done sequentially starting from the coarsest scale (smallest image) and going down the pyramid, updating your estimate as you go. It is very easy to implement by adding recursive calls to your original single-scale implementation. The running time of your implementation should be less than 1 minute."""

def colorize_image_recursively(b:np.array,g:np.array,r:np.array)->(np.array,list):
    """
     Align the high-resolution three color channel images. Return the colored image
     and a list of the displacement vector for each channel.

     :param b: the high-resolution blue channel image, which is a 3218x3741 numpy array.
     :param g: the high-resolution greeb channel image, which is a 3218x3741 numpy array.
     :param r: the high-resolution red channel image, which is a 3218x3741 numpy array.

     :return: a tuple of (colored_image, displacements). ''colored_image'' is a 3218x3741x3 numpy array.
     ''displacements'' is a list of the displacement vector for each channel.
    """
    colored_img=None
    displacements=[]

    # ADD YOUR CODE HERE (15 pts)
    # set the minimum size/threshold (224x224), stop the recursion at that point (base case)
    if (b.shape[0], g.shape[0], r.shape[0] <= 224):
      # do the displacement by calling the align_imga_to_imgb()
      aligned_r,dis_r = align_imga_to_imgb(r,g)
      aligned_b,dis_b = align_imga_to_imgb(b,g)
      aligned_g,dis_g = g,(0,0)

      colored_img = np.stack([aligned_r,aligned_b,aligned_g], axis=2)
      displacements = [dis_r, dis_b, dis_g]
      return colored_img, displacements

    # resize b,g,r separately by converting to an image and then back to np array
    new_r = Image.fromarray(r)
    new_b = Image.fromarray(b)
    new_g = Image.fromarray(g)

    new_r = r.resize(new_r.width//2, new_r.height//2)
    new_b = b.resize(new_b.width//2, new_b.height//2)
    new_g = g.resize(new_g.width//2, new_g.height//2)

    new_r = np.array(new_r)
    new_b = np.array(new_b)
    new_g = np.array(new_g)

    new_colored_img, new_displacements = colorize_image_recursively(new_b, new_g, new_r)

    dis_r = dis_r *2
    dis_b = dis_b *2
    dis_g = dis_g *2

    # shift the image again with np.roll
    shifted_r = np.roll(np.roll(new_r, dis_r, axis = 0), dis_r, axis = 1)
    shifted_b = np.roll(np.roll(new_b, dis_b, axis = 0), dis_b, axis = 1)
    shifted_g = new_g,(0,0)

    # align again?
    #aligned_r,dis_r = align_imga_to_imgb(new_r, new_g, 15)
    #aligned_b,dis_b = align_imga_to_imgb(new_b, new_g, 15)
    #aligned_g,dis_g = new_g,(0,0)

    colored_img = np.stack([shifted_r,shifted_g,shifted_b], axis=2)
    displacements = [dis_r, dis_g, dis_b]

    return colored_img,displacements

blue,green,red=crop_bgr_from_img('01047u.tif')
hrs_colored_img,hrs_displacements=colorize_image_recursively(blue,green,red)
print('Displacement of aligning the red channel to the green channel:',hrs_displacements[0])
print('Displacement of aligning the green channel to the green channel:',hrs_displacements[1])
print('Displacement of aligning the blue channel to the green channel:',hrs_displacements[2])
plt.figure(figsize=(40,40))
plt.axis('off')
plt.imshow(hrs_colored_img)

"""**d)** &ensp; **Improve the alignment [5 bonus pts].** The borders of the photograph will have strange colors since the three channels won't exactly align. See if you can devise an automatic way of cropping the border to get rid of the bad stuff. One possible idea is that the information in the good parts of the image generally agrees across the color channels, whereas at borders it does not."""

def crop_border(img:np.array)->np.array:
    """
     Crop the border to get rid of strange colors in the image.

     :param img: the colorized image, which is a 341x396x3 numpy array.
     :return: the improved image, which is a HxWx3 numpy array.

    """
    new_img=None

    # ADD YOUR CODE HERE

    return new_img

new_img=crop_border(colored_img)
plt.figure(figsize=(5,5))
plt.axis('off')
plt.imshow(new_img)
