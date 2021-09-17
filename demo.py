import cv2
import os
import glob
from skimage.util import random_noise
from imwatermark import WatermarkEncoder
from imwatermark import WatermarkDecoder
from blind_watermark import WaterMark
import numpy as np
import random
import glob
import pandas as pd

global word_length
global testfolder
global testattackfolder
global encodeoutputfolder
testfolder='test'
encodeoutputfolder = 'output'
testattackfolder='attack_output'

def encodeA(img,text):
    out_name='Algo_A_'+img
    out_path=os.path.join(testfolder,encodeoutputfolder,out_name)
    in_path=os.path.join(testfolder,img)

    encoder = WaterMark(password_img=1, password_wm=1)
    encoder.read_img(in_path)
    encoder.read_wm(text,mode='str')
    encoder.embed(out_path)

    print("Done Encoding Algorithm A (DWT): "+ out_path)



def encodeB(img,text):
    out_name='Algo_B_'+img
    out_path=os.path.join(testfolder,encodeoutputfolder,out_name)
    in_path=os.path.join(testfolder,img)

    encoder = WatermarkEncoder()

    in_img = cv2.imread(in_path)

    encoder.set_watermark('bytes', text.encode('utf-8'))

    img_encoded = encoder.encode(in_img,'dwtDct')

    cv2.imwrite(out_path,img_encoded)

    print("Done Encoding Algorithm B (DWTDCT): "+ out_path)



def encodeC(img,text):
    out_name='Algo_C_'+img
    out_path=os.path.join(testfolder,encodeoutputfolder,out_name)
    in_path=os.path.join(testfolder,img)

    encoder = WatermarkEncoder()

    in_img = cv2.imread(in_path)

    encoder.set_watermark('bytes', text.encode('utf-8'))

    img_encoded = encoder.encode(in_img,'dwtDctSvd')

    cv2.imwrite(out_path,img_encoded)

    print("Done Encoding Algorithm C (DWTDCTSVD): "+ out_path)



def encodeD(img,text):
    out_name='Algo_D_'+img
    out_path=os.path.join(testfolder,encodeoutputfolder,out_name)
    in_path=os.path.join(testfolder,img)

    if len(text) > 4 | len(text) <0 :
        print("Algorithm D(RivaGAN) Support 1-4 char length only.")
    else:
        encoder = WatermarkEncoder()

        in_img = cv2.imread(in_path)

        encoder.set_watermark('bytes', text.encode('utf-8'))
        encoder.loadModel()
        img_encoded = encoder.encode(in_img,'rivaGan')

        cv2.imwrite(out_path,img_encoded)

        print("Done Encoding Algorithm D (RivaGAN): "+ out_path)



def listdir(dirpath):        
    img_list=os.listdir(dirpath)
    return img_list

def decodeA(img_list): #pass in list
    word_length=int(input('Please enter char length expected: >'))
    bytes=(word_length*8)-1
    for img in img_list:
        in_path=os.path.join(testfolder,testattackfolder,img)
        decoder = WaterMark(password_img=1,password_wm=1)
        try:
            output = decoder.extract(in_path, wm_shape=bytes, mode='str')
            print("Decoded",img,": ", output)
        except:
            print("Decoded",img,": ", " Fail due to decode error")
        print("--------------------------")
    print("--------Finished Decoding---------")

def decodeB(img_list):
    word_length=int(input('Please enter char length expected: >'))
    bytes=(word_length*8)
    for img in img_list:
        in_path=os.path.join(testfolder,testattackfolder,img)
        bgr = cv2.imread(in_path)
        try:
            decoder = WatermarkDecoder('bytes', bytes)
            watermark = decoder.decode(bgr,'dwtDct')
            output = watermark.decode('utf-8')
            print("Decoded ",img,": ", output)
        except:
            print("Decoded ",img,": ", " Fail due to decode error")
        print("--------------------------")
    print("--------Finished Decoding---------")

def decodeC(img_list):
    word_length=int(input('Please enter char length expected: >'))
    bytes=(word_length*8)
    for img in img_list:
        in_path=os.path.join(testfolder,testattackfolder,img)
        bgr = cv2.imread(in_path)
        try:
            decoder = WatermarkDecoder('bytes', bytes)
            watermark = decoder.decode(bgr,'dwtDctSvd')
            output = watermark.decode('utf-8')
            print("Decoded ",img,": ", output)
        except:
            print("Decoded ",img,": ", " Fail due to decode error")
        print("--------------------------")
    print("--------Finished Decoding---------")

def decodeD(img_list):
    for img in img_list:
        in_path=os.path.join(testfolder,testattackfolder,img)
        bgr = cv2.imread(in_path)
        decoder = WatermarkDecoder('bytes', 32)
        decoder.loadModel()
        try:
            watermark = decoder.decode(bgr,'rivaGan')
            output = watermark.decode('utf-8')
            print("Decoded ",img,": ", output)
        except:
            print("Decoded ",img,": ", " Fail due to decode error")
        print("--------------------------")
    print("--------Finished Decoding---------")

def clear(): #exit program and purge all test files
    files = glob.glob(os.path.join(testfolder,testattackfolder,'*'))
    for f in files:
        os.remove(f)
        print("Removed ", f)

    files = glob.glob(os.path.join(testfolder,encodeoutputfolder,'*'))
    for f in files:
        os.remove(f)
        print("Removed ", f)
    print("Done clearing all test files.")


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

    output_path = os.path.join(testfolder, testattackfolder)

    gaussian(input_image, os.path.join(output_path, 'Gaussian_Noise.png'))
    salt_pepper(input_image, os.path.join(output_path, 'Salt_Pepper.png'))
    speckle(input_image, os.path.join(output_path, 'Speckle.png'))
    poisson(input_image, os.path.join(output_path, 'Poisson.png'))
    rotate(input_image, os.path.join(output_path, 'Rotate.png'), angle=10)
    upscale(input_image, os.path.join(output_path, 'ScaleUp.png'))
    downscale(input_image, os.path.join(output_path, 'ScaleDown.png'))
    averaging(input_image, os.path.join(output_path, 'Averaging.png'))
    gaussian_blurring(input_image, os.path.join(output_path, 'Gaussian_Blurring.png'))
    median_blurring(input_image, os.path.join(output_path, 'Median_Blurring.png'))
    bilateral_filtering(input_image, os.path.join(output_path, 'Bilateral_Filtering.png'))
    sharpen_filtering(input_image, os.path.join(output_path, 'Sharpen_Filtering.png'))
    crop_horizontal(input_image, os.path.join(output_path, 'Crop_Horizontal.png'))
    crop_vertical(input_image, os.path.join(output_path, 'Crop_Vertical.png'))
    increase_brightness(input_image, os.path.join(output_path, 'Brightness_Increase.png'))
    decrease_brightness(input_image, os.path.join(output_path, 'Brightness_Decrease.png'))
    masks(input_image, os.path.join(output_path, 'Masks.png'))
    cv2.imwrite(os.path.join(output_path, 'JPG.jpg'), input_image)

    print("Done Attacking.")

def get_img():
    dir_list = listdir(os.path.join(testfolder))
    final_list=[]
    print("Listing Images in Test Directory")
    n=1
    for img in dir_list:
        if img.endswith('.png'):
            print(n,". ", img)
            final_list.append(img)
            n+=1
    choice = int(input('Please choose the image: >'))

    if choice > len(final_list) or choice < 0:
        print("Image does not exist")

    print("Chosen Image: ",final_list[choice-1])

    return final_list[choice-1]

def get_atk_img(dir_list):
    print("Listing Images in Test Directory")
    n=1
    for img in dir_list:
        print(n,". ", img)
        n+=1
    choice = int(input('Please choose the image: >'))

    if choice > len(dir_list) or choice < 0:
        print("Image does not exist")

    print("Chosen Image: ",dir_list[choice-1])

    return dir_list[choice-1]

def attack_menu():
    print("Attacking Menu:")
    print("================")
    input_path=os.path.join(testfolder,encodeoutputfolder)
    img_list = listdir(input_path)
    img = get_atk_img(img_list)
    in_img = cv2.imread(os.path.join(input_path,img))
    init_attack(in_img, img)
    

def encode_menu():
    x=0
    while x != 5: 
        print("Encoding Watermark Into Images")
        print("=====================================")
        print("1. Algorithm A: DWT")
        print("2. Algorithm B: DWTDCT")
        print("3. Algorithm C: DWTDCTSVD")
        print("4. Algorithm C: RivaGan(4char max)")
        print("5. Exit\n\n")

        x = int(input("Please Input your choice: >"))

        if x == 1:
            print("Algorithm A: DWT")
            img = get_img()
            print("You have chosen :", img)
            in_text = input("Please enter a text to be encoded into the image: >")
            encodeA(img,in_text)

        elif x == 2:
            print("Algorithm B: DWTDCT")
            img = get_img()
            print("You have chosen :", img)
            in_text = input("Please enter a text to be encoded into the image: >")
            encodeB(img,in_text)
 
        elif x == 3:
            print("Algorithm C: DWTDCTSVD")
            img = get_img()
            print("You have chosen :", img)
            in_text = input("Please enter a text to be encoded into the image: >")
            encodeC(img,in_text)
        elif x == 4:
            print("Algorithm D: RivaGAN")
            img = get_img()
            print("You have chosen :", img)
            in_text = input("Please enter a text to be encoded into the image(4 char max): >")
            if len(in_text) <= 4:
                encodeD(img,in_text)
            else:
                print("Exceeded maximum char length")
        elif x == 5:
            print("exiting")
            break

        else:
            print("Wrong Input.")

def decode_menu():
    decode_path = os.path.join(testfolder,testattackfolder)
    img_list=listdir(decode_path)

    x=0
    while x != 5: 
        print("Decoding Watermark")
        print("=====================================")
        print("1. Algorithm A: DWT")
        print("2. Algorithm B: DWTDCT")
        print("3. Algorithm C: DWTDCTSVD")
        print("4. Algorithm C: RivaGan(4char max)")
        print("5. Exit\n\n")

        x = int(input("Please Input your choice: >"))

        if x == 1:
            print("Algorithm A: DWT")
            decodeA(img_list)

        elif x == 2:
            print("Algorithm B: DWTDCT")
            decodeB(img_list)
 
        elif x == 3:
            print("Algorithm C: DWTDCTSVD")
            decodeC(img_list)

        elif x == 4:
            print("Algorithm D: RivaGAN")
            decodeD(img_list)

        elif x == 5:
            print("exiting")
            break

        else:
            print("Wrong Input.")

def menu():
    x=0
    while x != 5: 
        print("Demo Tools For Invisible Watermarking")
        print("=====================================")
        print("1. Encode")
        print("2. Attack Images")
        print("3. Decode")
        print("4. Clear Test")
        print("5. Exit\n\n")

        x = int(input("Please Input your choice: >"))

        if x == 1:
            encode_menu()
        elif x == 2:
            attack_menu()
        elif x == 3:
            decode_menu()
        elif x == 4:
            clear()
        elif x == 5:
            clear()
            print("exiting")
            break

        else:
            print("Wrong Input.")


def main():
    menu()
    print("System stopped.")
  


if __name__ == "__main__":
    main()