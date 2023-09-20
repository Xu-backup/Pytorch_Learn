import numpy
import numpy as np
import math
import cv2
import torch
import numpy as np
from PIL import Image
from scipy.signal import convolve2d
from torch.autograd import Variable



def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)

def ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):
    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    M, N = im1.shape
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    window = matlab_style_gauss2D(shape=(win_size, win_size), sigma=1.5)
    window = window / np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1 * im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2 * im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1 * im2, window, 'valid') - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigmal2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return np.mean(np.mean(ssim_map))

def psnr(img1, img2):
    img1 = np.float64(img1)[:,:]
    img2 = np.float64(img2)[:,:]
    mse = numpy.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

if __name__ == '__main__':
    total = 0
    for i in range(0,32):
        original = cv2.imread(f"D:/Research/Reproduction/reproduction_save/target{i}-0.png", 1)  # numpy.adarray
        contrast = cv2.imread(f"D:/Research/Reproduction/reproduction_save/{i}-0.png", 1)

        ssim_original = np.array(Image.open(f"D:/Research/Reproduction/reproduction_save/target{i}-0.png"))[:, :, 0]
        ssim_contrast = np.array(Image.open(f"D:/Research/Reproduction/reproduction_save/{i}-0.png"))[:, :, 0]
    # print("psnr",psnr(original,contrast))
    #     if ssim(ssim_original,ssim_contrast) > 0.5:
        total += ssim(ssim_original,ssim_contrast)
        print(f"ssim{i}",ssim(ssim_original,ssim_contrast))
    print(total/32)