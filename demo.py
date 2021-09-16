import cv2
import cv2
import os
from skimage.util import random_noise
from imwatermark import WatermarkEncoder
from imwatermark import WatermarkDecoder
import numpy as np
import random
import glob
import pandas as pd


def encodeA():


def encodeB():


def encodeC():


def encodeD():


def decodeA():


def decodeB():


def decodeC():


def decodeD():



    #####################Attacking Method#############

    # Attacking Methods

def gaussian(img, output_file_name):
    img_shape = img.shape
    gauss_img = random_noise(img, mode='gaussian', mean=0, var=0.01, clip=True)
    gauss_img = np.array(255*gauss_img, dtype='uint8')
    cv2.imwrite(output_file_name, gauss_img)


def salt_pepper(img, output_file_name):
    ratio = 0.1  # Probability of the noise
    # 0.01 all pass?
    # 0.1 algorithm d only partial pass
    snp_img = np.zeros(img.shape, np.uint8)
    thres = 1 - ratio
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rdn = random.random()
            if rdn < ratio:
                snp_img[i][j] = 0
            elif rdn > thres:
                snp_img[i][j] = 255
            else:
                snp_img[i][j] = img[i][j]
    cv2.imwrite(output_file_name, snp_img)


def speckle(img, output_file_name):
    speckle_img = random_noise(
        img, mode='speckle', mean=0, var=0.01, clip=True)
    # The above function returns a floating-point image on the range [0, 1], thus we changed it to 'uint8' and from [0,255]
    speckle_img = np.array(255*speckle_img, dtype='uint8')
    cv2.imwrite(output_file_name, speckle_img)


def poisson(img, output_file_name):
    img_shape = img.shape
    noise = np.random.poisson(20, img_shape)
    # 30 all fail
    output = img + noise
    cv2.imwrite(output_file_name, output)


def rotate(img, output_file_name, angle=10):
    rows, cols, _ = img.shape
    M = cv2.getRotationMatrix2D(
        center=(cols / 2, rows / 2), angle=angle, scale=1)
    output_img = cv2.warpAffine(img, M, (cols, rows))
    cv2.imwrite(output_file_name, output_img)


def upscale(img, output_file_name):
    img_shape = img.shape
    output_img = cv2.resize(img, None, fx=1.25, fy=1.25,
                            interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(output_file_name, output_img)


def downscale(img, output_file_name):
    img_shape = img.shape
    output_img = cv2.resize(img, None, fx=0.75, fy=0.75,
                            interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(output_file_name, output_img)


def averaging(img, output_file_name):
    img_shape = img.shape
    output_img = cv2.blur(img, (5, 5))
    cv2.imwrite(output_file_name, output_img)


def gaussian_blurring(img, output_file_name):
    img_shape = img.shape
    output_img = cv2.GaussianBlur(img, (5, 5), 0)
    cv2.imwrite(output_file_name, output_img)


def median_blurring(img, output_file_name):
    output_img = cv2.medianBlur(img, 7)
    # 5 all pass
    cv2.imwrite(output_file_name, output_img)


def bilateral_filtering(img, output_file_name):
    output_img = cv2.bilateralFilter(img, 9, 75, 75)
    cv2.imwrite(output_file_name, output_img)


def sharpen_filtering(img, output_file_name):
    sharpen_filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharped_img = cv2.filter2D(img, -1, sharpen_filter)
    cv2.imwrite(output_file_name, sharped_img)


def crop_horizontal(img, output_file_name):
    ratio = 0.5
    # left 50% all pass
    img_shape = img.shape
    height = int(img_shape[0] * ratio)
    horizontal_img = img[:height, :, :]
    cv2.imwrite(output_file_name, horizontal_img)


def crop_vertical(img, output_file_name):
    ratio = 0.5
    img_shape = img.shape
    vertical = int(img_shape[1] * ratio)
    vertical_img = img[:, :vertical, :]
    cv2.imwrite(output_file_name, vertical_img)


def increase_brightness(img, output_file_name):
    ratio = 1.4
    # +10% all pass
    # +50% algorithm D
    inc_bright_img = img * ratio
    inc_bright_img[inc_bright_img > 255] = 255
    cv2.imwrite(output_file_name, inc_bright_img)


def decrease_brightness(img, output_file_name):
    ratio = 0.6
    # -50% brightness all pass
    dec_bright_img = img * ratio
    dec_bright_img[dec_bright_img > 255] = 255
    cv2.imwrite(output_file_name, dec_bright_img)


def masks(img, output_file_name):
    n = 5  # Set 2 mask
    ratio = 0.3  # mask ratio
    input_img_shape = img.shape
    mask_img = img.copy()
    for i in range(n):
        # random one place to put mask，1-ratio to avoid overfloat
        tmp = np.random.rand() * (1 - ratio)
        start_height, end_height = int(
            tmp * input_img_shape[0]), int((tmp + ratio) * input_img_shape[0])
        tmp = np.random.rand() * (1 - ratio)
        start_width, end_width = int(
            tmp * input_img_shape[1]), int((tmp + ratio) * input_img_shape[1])

        mask_img[start_height:end_height, start_width:end_width, :] = 0
    cv2.imwrite(output_file_name, mask_img)


def init_attack(input_image, image_name):
    print('Attacking ' + image_name)

    a = image_name.split('.')[0]
    mri_name = a.split('_')[2]+'_'+a.split('_')[3]
    algo_name = a.split('_')[0]+'_'+a.split('_')[1]

    output_path = os.path.join('out_att', mri_name, algo_name.lower())

    gaussian(input_image, os.path.join(output_path, 'Gaussian_Noise.png'))
    salt_pepper(input_image, os.path.join(output_path, 'Salt_Pepper.png'))
    speckle(input_image, os.path.join(output_path, 'Speckle.png'))
    poisson(input_image, os.path.join(output_path, 'Poisson.png'))
    rotate(input_image, os.path.join(output_path, 'Rotate.png'), angle=10)
    upscale(input_image, os.path.join(output_path, 'ScaleUp.png'))
    downscale(input_image, os.path.join(output_path, 'ScaleDown.png'))
    averaging(input_image, os.path.join(output_path, 'Averaging.png'))
    gaussian_blurring(input_image, os.path.join(
        output_path, 'Gaussian_Blurring.png'))
    median_blurring(input_image, os.path.join(
        output_path, 'Median_Blurring.png'))
    bilateral_filtering(input_image, os.path.join(
        output_path, 'Bilateral_Filtering.png'))
    sharpen_filtering(input_image, os.path.join(
        output_path, 'Sharpen_Filtering.png'))
    crop_horizontal(input_image, os.path.join(
        output_path, 'Crop_Horizontal.png'))
    crop_vertical(input_image, os.path.join(output_path, 'Crop_Vertical.png'))
    increase_brightness(input_image, os.path.join(
        output_path, 'Brightness_Increase.png'))
    decrease_brightness(input_image, os.path.join(
        output_path, 'Brightness_Decrease.png'))
    masks(input_image, os.path.join(output_path, 'Masks.png'))
    cv2.imwrite(os.path.join(output_path, 'JPG.jpg'), input_image)
