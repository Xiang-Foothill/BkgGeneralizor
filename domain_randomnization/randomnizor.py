import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
import os
import torch
from PIL import Image

BKGSET_PATH = os.path.dirname(os.path.abspath(__file__)) + "/Bkgset.pth"

class BkgRandomnizer():
    def __init__(self, transfer_percentage, debug = False, bkgSet_path = BKGSET_PATH):
          self.transfer_percentage = transfer_percentage
          self.background_set = []
          self.write_field = "camera"
          self.debug = debug
          self.bkgSet_path = bkgSet_path
          
          try:
            self.bkgset = torch.load(self.bkgSet_path)
            logger.info(f"//////////////////////////////////////////// The background set is loaded, it already has {self.bkgset[0].shape[0]} pictures //////////////////////////////")
          except:
            logger.info(f"No bacground image set found in the directory {self.bkgSet_path}")
              
          # Temporary: built-in rendering
          if self.debug:
            plt.ion()  # Turn on interactive mode
            self.fig, self.ax = plt.subplots()
            self.display_frame = self.ax.imshow(np.zeros((64, 64, 3), dtype=np.uint8))
            plt.show()

    def randomnize(self, input_image, input_mask):
        setSize = self.bkgset.shape[0]
        idx = np.random.randint(low = 0, high = setSize)

        bkg = self.bkgset[idx]

        # rescale the background image so that it has the same shape as the input image
        bkg_pil = Image.fromarray(bkg)  # Convert to PIL Image
        bkg_resized = bkg_pil.resize((input_image.shape[0], input_image.shape[0]))  # Resize to [H, H]
        bkg = np.array(bkg_resized)  # Convert back to NumPy array

        return input_image * (1 - input_mask) + bkg * input_mask
    
    def null_randomnize(self, input_image, input_mask):
        """a null function used to test the interface"""

        return input_image * (1 - input_mask)

    def change_obs(self, obs):
        if np.random.uniform(low = 0.0, high = 1.0) <= self.transfer_percentage:
              obs[self.write_field] = self.randomnize(input_image = obs["camera"], input_mask = obs["semantics"])
        
        if self.debug:
            self.display_frame.set_data(obs[self.write_field])  # <- update image data
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

