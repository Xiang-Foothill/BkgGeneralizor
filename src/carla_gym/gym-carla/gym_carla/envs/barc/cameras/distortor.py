import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
from mpclab_common.pytypes import VehicleState

"""A set of functions used to add a certain kind of distortion effects  to the input image"""

def gaussian_noise(image, state = None, mean = 0, sigma = 15.0):
    """add guassian noise to the input image"""
    size = image.shape
    noise = np.random.normal(loc = mean, scale = sigma, size = size)
    res_image = np.clip(image + noise, a_min = 0, a_max = 255)
    res_image = res_image.astype(np.uint8)

    return res_image

def motion_blur(image, state, ksize = 20, w_scale = 0.02, vlong_scale = 0.05):
    """add bluring effect to the image based on the motion of the vehicle"""
    v_long = state.v.v_long
    w_psi = state.w.w_psi

    motion_y = v_long / w_scale
    motion_x = w_psi / vlong_scale

    blur_kernel = np.zeros((ksize, ksize))
    c = int(ksize/2)

    motion_x_mag, motion_y_mag = max(0, min(c - 1, abs(motion_x))), max(0,  min(c - 1, abs(motion_y)))
    motion_x, motion_y = int(motion_x_mag * np.sign(motion_x)), int(motion_y_mag * np.sign(motion_y))
    blur_kernel = cv2.line(blur_kernel, (c+motion_x,c+motion_y), (c,c), (1.0,), thickness = 1)
    normalizer = np.sum(blur_kernel > 0.5)
    blur_kernel = blur_kernel / normalizer

    blurred = np.zeros_like(image)
    for channel in range(image.shape[2]):
        blurred[:, :, channel] = cv2.filter2D(image[:, :, channel], -1, blur_kernel)

    return blurred

def overExposure(image, state = None, factor = 1.5):
    # Boost brightness (overexposure factor)
    overexposed = image * factor

    # Clip values to [0, 255]
    overexposed = np.clip(overexposed, 0, 255).astype(np.uint8)

    return overexposed

def underExposure(image, state = None, factor = 3.8):
    underexposed = image / factor

    # Clip values to [0, 255]
    underexposed = np.clip(underexposed, 0, 255).astype(np.uint8)

    return underexposed

def original_image(image, state = None):

    return image

"""functions below are test functions"""

def test_overExposure(test_set):
    random_idx = np.random.randint(low = 0.0, high = len(test_set))
    print(test_set[random_idx].shape)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(test_set[random_idx])
    ax2.imshow(underExposure(test_set[random_idx]))
    plt.show()

def test_guassian_noise(test_set):
    random_idx = np.random.randint(low = 0.0, high = len(test_set))
    print(test_set[random_idx].shape)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(test_set[random_idx])
    ax2.imshow(gaussian_noise(test_set[random_idx]))
    plt.show()

def test_motion_blur(test_set):
    w =  -1.0
    vx = 0.0
    state = VehicleState()
    state.v.v_long = vx
    state.w.w_psi = w

    random_idx = np.random.randint(low = 0.0, high = len(test_set))
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(test_set[random_idx])
    ax2.imshow(motion_blur(test_set[random_idx], state))
    plt.show()

if __name__ == "__main__":
    test_set = np.load("test_images.npy")
    test_overExposure(test_set)