import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

class BkgRandomnizer():
    def __init__(self, transfer_percentage, debug = False):
          self.transfer_percentage = transfer_percentage
          self.background_set = []
          self.write_field = "augmented"
          self.debug = debug

          # Temporary: built-in rendering
          if self.debug:
            plt.ion()  # Turn on interactive mode
            self.fig, self.ax = plt.subplots()
            self.display_frame = self.ax.imshow(np.zeros((64, 64, 3), dtype=np.uint8))
            plt.show()

    def randomnize(self, input_image, input_mask):
        raise NotImplementedError
    
    def null_randomnize(self, input_image, input_mask):
        """a null function used to test the interface"""

        return input_image * (1 - input_mask)

    def change_obs(self, obs):
        if np.random.uniform(low = 0.0, high = 1.0) <= self.transfer_percentage:
              obs[self.write_field] = self.null_randomnize(input_image = obs["camera"], input_mask = obs["semantics"])
        
        if self.debug:
            self.display_frame.set_data(obs[self.write_field])  # <- update image data
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

