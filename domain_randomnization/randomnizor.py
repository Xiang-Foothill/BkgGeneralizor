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
    def __init__(self, transfer_percentage, debug = False, no_background = False):
          self.transfer_percentage = transfer_percentage
          self.write_field = "camera"
          self.debug = debug
          self.bkgSet_path = random.choice(BKGSET_PATHS)
          self.switch_limit = 10000 # after doing randomization for such amount of times, switch to antoher BKG_SET to ensure that the background is diverse enough
          self.cur_counts = 0

          # Temporary: built-in rendering
          if self.debug:
             plt.ion()  # Turn on interactive mode
             self.fig, self.ax = plt.subplots()
             self.display_frame = self.ax.imshow(np.zeros((64, 64, 3), dtype=np.uint8))
             plt.show()

          if no_background:
             return
          
          try:
              self.bkgset = torch.load(self.bkgSet_path)
              self.bkgset = self.bkgset.cpu().numpy()
              logger.info(f"//////////////////////////////////////////// The background set at {self.bkgSet_path} is loaded, it already has {self.bkgset.shape[0]} pictures //////////////////////////////")
          except:
              logger.info(f"No bacground image set found in the directory {self.bkgSet_path}")
            
          try:
              self.testSet = torch.load(TESTSET_PATH)
              self.testSet = self.testSet.cpu().numpy()
              logger.info(f"//////////////////////////////////////////// The background set at {TESTSET_PATH} is loaded, it already has {self.testSet.shape[0]} pictures //////////////////////////////")
          except:
              logger.info(f"No bacground image set found in the directory {TESTSET_PATH}")
          
          #try:
          #     self.textureSet = torch.load(TEXTURE_PATH)
          #     self.textureSet = self.textureSet.cpu().numpy()
          #     logger.info(f"//////////////////////////////////////////// The texture set at {TEXTURE_PATH} is loaded, it already has {self.textureSet.shape[0]} pictures //////////////////////////////")
          # except:
          #     logger.info(f"No bacground image set found in the directory {TEXTURE_PATH}")

    def bkg_randomnize(self, input_image, image_set, **kwargs):
        input_mask = kwargs["semantics"]

        setSize = image_set.shape[0]
        idx = np.random.randint(low = 0, high = setSize)

        bkg = image_set[idx]

        # rescale the background image so that it has the same shape as the input image
        bkg_pil = Image.fromarray(bkg)  # Convert to PIL Image
        bkg_resized = bkg_pil.resize((input_image.shape[0], input_image.shape[0]))  # Resize to [H, H]
        bkg = np.array(bkg_resized)  # Convert back to NumPy array

        return input_image * (1 - input_mask) + bkg * input_mask
    
    def change_road_color(self, image, **kwargs):
      """
      Augments the road area in the image with a randomized grey/black color and artificial cracks.

      Parameters:
      - image: np.ndarray of shape (H, W, 3), RGB image.
      - road_mask_rgb: np.ndarray of shape (H, W, 3), binary road mask (white=road, black=non-road).

      Returns:
      - Augmented image as np.ndarray of shape (H, W, 3).
      """
      road_mask_rgb = 1 - kwargs["semantics"]
      assert image.shape == road_mask_rgb.shape, "Image and mask shape must match."

      # Convert RGB road mask to binary [H, W] mask
      road_mask = (np.all(road_mask_rgb == 255, axis=-1)).astype(np.uint8)

      # Predefined set of asphalt-like black/grey tones (discrete and realistic)
      asphalt_colors = [
        (30, 30, 30),
        (45, 45, 45),
        (60, 60, 60),
        (75, 75, 75),
        (90, 90, 90),
        (105, 105, 105),
        (120, 120, 120),
        (40, 40, 40),
        (80, 80, 80),
        (100, 100, 100)
      ]

      # Randomly select one color
      selected_color = np.array(random.choice(asphalt_colors), dtype=np.uint8)

      # Apply the color to the road region
      augmented = image.copy()
      augmented = augmented * (1 - road_mask_rgb) + road_mask_rgb * selected_color

      return augmented

    def road_randomnize(self, input_image, **kwargs):
       
      input_mask = 1 - kwargs["semantics"]
      setSize = self.textureSet.shape[0]
      idx = np.random.randint(low = 0, high = setSize)
      texture = self.textureSet[idx]

      # rescale the background image so that it has the same shape as the input image
      texture_pil = Image.fromarray(texture)  # Convert to PIL Image
      texture_resized = texture_pil.resize((input_image.shape[0], input_image.shape[0]))  # Resize to [H, H]
      texture = np.array(texture_resized)  # Convert back to NumPy array

      return input_image * (1 - input_mask) + texture * input_mask
       
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

    def randomnize(self, data):
        """
        The interface to the fetch function of the data_loader
        put all the available randomnize functions here
        @ data: a dictionary for set of data retrieved from the replay_buffer
        @ return: an augmented RGB image"""

        image = np.copy(data["camera"])
        aux_augments = [self.gaussian_noise, self.overExposure, self.underExposure, self.random_cutouts, self.gaussian_blur]
        aux_augment = random.choice(aux_augments)

        if np.random.uniform(low = 0.0, high = 1.0) < self.transfer_percentage:
          image = self.bkg_randomnize(input_image=image, image_set = self.bkgset, **data)
          
          # check whether to switch to another bkgset or not
          self.cur_counts = self.cur_counts + 1
          if self.cur_counts >= self.switch_limit:
             self.cur_counts = 0
             logger.info("Hit the switch limit, Loading another Bkgset")
             self.bkgSet_path = random.choice(BKGSET_PATHS)
             self.bkgset = torch.load(self.bkgSet_path)
             self.bkgset = self.bkgset.cpu().numpy()
             logger.info(f"//////////////////////////////////////////// The background set at {self.bkgSet_path} is loaded, it already has {self.bkgset.shape[0]} pictures //////////////////////////////")
        
        if np.random.uniform(low = 0.0, high = 1.0) < self.transfer_percentage:
           image = self.change_road_color(image, **data)

        if np.random.uniform(low = 0.0, high = 1.0) < self.transfer_percentage:
          image = aux_augment(image = image, **data)
        
        if self.debug:
            self.display_frame.set_data(image)  # <- update image data
            time.sleep(0.5)
            
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        return image
    
    def traditional_randomnize(self, data):
      image = np.copy(data["camera"])
      aux_augments = [self.gaussian_noise, self.overExposure, self.underExposure, self.random_cutouts, self.gaussian_blur]
      aux_augment = random.choice(aux_augments)

      if np.random.uniform(low = 0.0, high = 1.0) < self.transfer_percentage:
          image = aux_augment(image = image, **data)
      
      if self.debug:
          self.display_frame.set_data(image)  # <- update image data
          time.sleep(0.5)
            
          self.fig.canvas.draw()
          self.fig.canvas.flush_events()

      return image

    def null_randomnize(self, input_image, input_mask):
        """a null function used to test the interface"""
        return input_image * (1 - input_mask)

    def change_obs(self, obs, ifTest = False):
        """the interface to the BARC_ENV, it will change the observation from the environment directly"""

        if np.random.uniform(low = 0.0, high = 1.0) <= self.transfer_percentage:
            if ifTest:
                image_set = self.testSet
            else:
                image_set = self.bkgset

            obs[self.write_field] = self.bkg_randomnize(input_image = obs["camera"], image_set = image_set, **obs)
        
        if self.debug:
            self.display_frame.set_data(obs[self.write_field])  # <- update image data
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

class ContrastRandomnizer(BkgRandomnizer):
    def __init__(self):
      super().__init__(transfer_percentage=1.0, debug = False)

    def negatives_randomnize(self, data):
      # to be optimzied: add randomization to the negative examples
      return data["negatives"]

class linProgRandomnizer(BkgRandomnizer):
  """progressive randomizer that gradualy increases the amount of images being domain-randomized during training"""

  def __init__(self, final_percent, no_background = False, debug = False, mode = "constant"):
      """mode constant: the randomizer will behave exactly the same as BkgRandomnizer
      mode linear: the transfer_percentage will increase constantly"""
      super().__init__(transfer_percentage = 0.0, debug = debug, no_background = no_background)
      self.final_percent = final_percent
      self.mode = mode # choose either "constant" or "linear"
  
  def update_cur(self, global_step, total_epochs):
    if self.mode == "constant":
      self.transfer_percentage = self.final_percent
    elif self.mode == "linear":
      self.transfer_percentage = self.final_percent * (global_step + 1) / total_epochs
    logger.info(f"curriculum updated: the transfer percentage right now is {self.transfer_percentage}")
  


   
        

