import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
import os
import torch
from PIL import Image
import time
import random
import cv2

# BKGSET_PATH = os.path.dirname(os.path.abspath(__file__)) + "/Bkgset.pth"
BKGSET_PATH1 = '../Backgrounds/Bkgset1.pth'
BKGSET_PATH2 = '../Backgrounds/Bkgset2.pth'
BKGSET_PATH3 = '../Backgrounds/Bkgset3.pth'
BKGSET_PATH4 = '../Backgrounds/Bkgset4.pth'
BKGSET_PATH5 = '../Backgrounds/Bkgset5.pth'
BKGSET_PATH6 = '../Backgrounds/Bkgset6.pth'
BKGSET_PATHS = [BKGSET_PATH1, BKGSET_PATH2, BKGSET_PATH3, BKGSET_PATH4, BKGSET_PATH5, BKGSET_PATH6]
TESTSET_PATH = '../Backgrounds/testSet.pth'
TEXTURE_PATH = '../Backgrounds/textureSet.pth'

class BkgRandomnizer():
    def __init__(self, transfer_percentage):
      self.transfer_percentage = transfer_percentage
      
    def gaussian_noise(self, image, mean = 0, sigma = 15.0, **kwargs):

      """add guassian noise to the input image"""
      size = image.shape
      noise = np.random.normal(loc = mean, scale = sigma, size = size)
      res_image = np.clip(image + noise, a_min = 0, a_max = 255)
      res_image = res_image.astype(np.uint8)

      return res_image
    
    def overExposure(self, image, max_factor = 1.5, min_factor = 1.1, **kwargs):
      factor = np.random.uniform(low = min_factor, high = max_factor)
      # Boost brightness (overexposure factor)
      overexposed = image * factor

      # Clip values to [0, 255]
      overexposed = np.clip(overexposed, 0, 255).astype(np.uint8)

      return overexposed
    
    def underExposure(self, image, min_factor = 1.5, max_factor = 2.5, **kwargs):
      factor = np.random.uniform(low = min_factor, high = max_factor)
      underexposed = image / factor

      # Clip values to [0, 255]
      underexposed = np.clip(underexposed, 0, 255).astype(np.uint8)

      return underexposed
    
    def random_cutouts(self, image, max_cutout_area_ratio=0.08, max_rects=2, **kwargs):
      """
      Applies random rectangular cutouts to the input image.
    
      Parameters:
        image (np.ndarray): Input image of shape (H, W, 3).
        max_cutout_area_ratio (float): Maximum total area to cut out, as a ratio of image area (0-1).
        max_rects (int): Maximum number of cutout rectangles to attempt.

      Returns:
        np.ndarray: Image with cutouts applied.
      """
      H, W, _ = image.shape
      total_area = H * W
      max_cutout_area = max_cutout_area_ratio * total_area
      cutout_area = 0

      image_cutout = image.copy()

      for _ in range(max_rects):
          if cutout_area >= max_cutout_area:
              break

          # Random width and height
          max_rect_area = max_cutout_area - cutout_area
          max_w = min(W, int(np.sqrt(max_rect_area)))
          max_h = min(H, int(np.sqrt(max_rect_area)))
          w = np.random.randint(1, max_w + 1)
          h = np.random.randint(1, max_h + 1)

          # Make sure new area does not exceed remaining allowed area
          rect_area = w * h
          if cutout_area + rect_area > max_cutout_area:
              continue

          # Random position
          x0 = np.random.randint(0, W - w + 1)
          y0 = np.random.randint(0, H - h + 1)

          # Apply cutout (black rectangle)
          image_cutout[y0:y0 + h, x0:x0 + w] = 0
          cutout_area += rect_area

      return image_cutout
    
    def gaussian_blur(self, image, max_blur_radius=4, **kwargs):
      """
      Applies a random Gaussian blur to the image.

      Parameters:
        image (np.ndarray): Input image of shape (H, W, 3).
        max_blur_radius (int): Maximum blur radius (must be >= 1).

      Returns:
        np.ndarray: Blurred image.
      """
      if max_blur_radius < 1:
        raise ValueError("max_blur_radius must be at least 1")

      # Choose a random odd kernel size between 1 and max_blur_radius*2 + 1
      max_kernel_size = max_blur_radius * 2 + 1
      kernel_size = np.random.choice([k for k in range(1, max_kernel_size + 1, 2)])

      blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
      return blurred_image
    
    def traditional_randomnize(self, data):
      image = np.copy(data["camera"])
      aux_augments = [self.gaussian_noise, self.overExposure, self.underExposure, self.random_cutouts, self.gaussian_blur]
      aux_augment = random.choice(aux_augments)

      if np.random.uniform(low = 0.0, high = 1.0) < self.transfer_percentage:
          image = aux_augment(image = image, **data)

      return image


   
        

