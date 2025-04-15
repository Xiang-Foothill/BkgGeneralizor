import numpy as np
from mpclab_common.pytypes import VehicleState, Position, OrientationEuler
import os
import pickle
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from PIL import Image
import random
import sys
from pathlib import Path
from tqdm import tqdm

X_MAX = 5.0
X_MIN = - 3.0
Y_MAX = 6.0
Y_MIN = -1.0
PSI_MAX = 5.5
PSI_MIN = - 2.0
DATA_PATH = os.path.expanduser("~/data/barc_data")

def crop_to_aspect_ratio(image, target_ratio):
    """Crops an image to a specific aspect ratio.

    Args:
        image: The input image (NumPy array).
        target_ratio: The desired aspect ratio (width / height).

    Returns:
        The cropped image.
    """

    height, width = image.shape[:2]
    current_ratio = width / height

    if current_ratio > target_ratio:
        # Image is wider than desired, crop the sides
        new_width = int(height * target_ratio)
        x_start = int((width - new_width) / 2)
        x_end = x_start + new_width
        cropped_image = image[:, x_start:x_end]
    else:
        # Image is taller than desired, crop the top and bottom
        new_height = int(width / target_ratio)
        y_start = int((height - new_height) / 2)
        y_end = y_start + new_height
        cropped_image = image[y_start:y_end, :]

    return cropped_image

class BaseCamera:
    def __init__(self, track_name = None, height = 480, width = 640, load_mode = False, data_path=DATA_PATH, host='localhost', port=2000):
        """
        @load_mode: a boolean variable
        if it equals to true, load an existing model saved in the directory load_path, 
        it equals to false, create an empty basecamera object waiting to be trained
        
        @load_path: the path from which you want to load your existing baseCamera object, if load_mode is equal to false, set this path to None"""

        self._height = height
        self._width = width
        # self.ratio = width / height
        self.camera_img = np.empty((height, width, 3), dtype=np.uint8)

        self.CACHE_SIZE = 1500 # specify the size of CACHE here
        self.CACHE = {} # a cache dicionary used for saving images that have been read, all these images might be reused in the future

        self.image_labels = {}

        if not os.path.exists(os.path.join(data_path, 'export')):
            self.train(data_path, save_path = data_path, model_name = "export")

        # if load_mode == False: # train the model with data saved in path

            # we treat state grids as data points and images as corresponding label to these data points
            # self.image_labels = None
            # self.finder = None
            # self.image_labels = {}

            # normalization vectors
            # self.state_average = None
            # self.state_std = None
        load_path = os.path.join(data_path, 'export')
        # else: # load the camera from the given LOAD_PATH
        with open(os.path.join(load_path, "finder.pkl"), "rb") as f:
            self.finder = pickle.load(f)
        with open(os.path.join(load_path, "image_labels.pkl"), "rb") as f:
            self.image_labels = pickle.load(f)
        # with open(os.path.join(load_path, "model.pkl"), "rb") as f:
        #     model = pickle.load(f)

        # self.finder = model.finder
        # self.image_labels = model.image_labels
        self.image_folder = os.path.join(load_path, "images")

        # Temporary: built-in rendering
        self.fig, self.ax = plt.subplots()
        self.im = self.ax.imshow(self.camera_img)

    @property
    def height(self) -> int:
        """the height of the camera image"""
        return self._height

    @property
    def width(self) -> int:
        """the width of the camera image"""
        return self._width
    
    def train(self, data_path, save_path, model_name, intervals = [0.2, 0.2, np.pi / 10], max_values = [X_MAX, Y_MAX, PSI_MAX], min_values = [X_MIN, Y_MIN, PSI_MIN]):

        """@data_path: the directory in which the data files used for training are saved
        @save_path: the directory in which the model is going to be saved
        @model_name: a string for the name of the model
        @max_values: [maximum value of x, maximum value of y, maximum value of psi]
        @min_values: [minimum value of x, minimum value of y, minimum value of psi]
        @intervals: accuracy of the state grid"""

        model_folder = os.path.join(save_path, model_name)
        image_folder = os.path.join(model_folder, "images")
        os.makedirs(model_folder, exist_ok=True)
        os.makedirs(image_folder, exist_ok=True)
        self.image_folder = image_folder # save the directory for the image folder as a global class variable for the convenience of future use

        self.make_finder(max_values, min_values, intervals)

        """iterate through all the data files one by one and complete the training process"""
        npzfileNames = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]

        # setup the progress toolbar
        toolbar_width = 150

        for npz_file_name in npzfileNames:
            # train the model with the files one by one
            npz_file = os.path.join(data_path, npz_file_name)
            data = np.load(npz_file, allow_pickle = True)
            states, images = data["states"][ : , 3 : ], data["images"]
            states[:, 2] = np.mod(states[:, 2], np.pi * 2)

            self.image_prepare(states, images, npz_file_name)

            # update the bar
        
        # with open(os.path.join(model_folder, "model.pkl"), "wb") as f:
        #     pickle.dump(self, f)
        with open(os.path.join(model_folder, "finder.pkl"), "wb") as f:
            pickle.dump(self.finder, f)
        with open(os.path.join(model_folder, "image_labels.pkl"), "wb") as f:
            pickle.dump(self.image_labels, f)

        sys.stdout.write("]\n") # this ends the progress bar

    def normalize(self, states, init_parameter = False):
        """normalize the states matrix
        @states: the state matrix to be normalized
        @init_parameter: True if the result of this normalization is used to update the class parameters, false otherwise
        if you are using this functin for the first time, make sure to set init_parameter to True"""

        if init_parameter:
            self.state_average = np.average(states, axis = 0)
            self.state_std = np.std(states, axis = 0)
        
        if self.state_std is None or self.state_average is None:
            raise ValueError("The self.state_std or self.state_average are None. Normalization cannot be implemented. If this function is called for the first time, make sure set the init_paramter as True")
        
        states = (states - self.state_average) / self.state_std

        return self.state_average, self.state_std
    
    def image_prepare(self, states, images, npz_file_name):
        if self.finder is None:
            raise ValueError("The finder is None, make sure to call make_finder before you call this function")
        
        dist, index_array = self.finder.kneighbors(states, n_neighbors = 1)
        index_array = index_array.flatten()

        for i, label_index in tqdm(enumerate(index_array), total=len(index_array), desc=npz_file_name):

            if label_index not in self.image_labels:
                image_token = npz_file_name + str(i) +".jpg"
                self.image_labels[label_index] = image_token
                pillow_image = Image.fromarray(np.moveaxis(images[i], [0, 1, 2], [2, 0, 1]))
                pillow_image.save(os.path.join(self.image_folder, image_token))

    def make_finder(self, max_values, min_values, intervals):
        """use sklearn NearestNeighbor model to cluster all the discrete grid states
        such a model will be later used to relate real states to their closest discrete grid states"""
        #read the parameters
        x_min, y_min, psi_min = min_values
        x_max, y_max, psi_max = max_values
        x_interval, y_interval, psi_interval = intervals

        # cut off the grid
        X, Y, PSI = np.mgrid[x_min : x_max : x_interval, y_min : y_max : y_interval, psi_min : psi_max : psi_interval]
        grid = np.vstack((X.flatten(), Y.flatten(), PSI.flatten())).T

        # prepare the nearest neighbor model
        neigh = NearestNeighbors(n_neighbors = 1)
        neigh.fit(grid)
        self.finder = neigh
    
    def check_grid(self, grid):
        """this is a function used for debugging the correctness of grid"""
        for i, row1 in enumerate(grid):
            if i < grid.shape[0] - 1 and i % 1000 == 0:
                for row2 in enumerate(grid[i + 1 : ]):
                    if np.array_equal(row1, row2):
                        print("repeated row!!!")

        print("this grid is strictly defined") 
    
    def query_rgb(self, state: VehicleState, max_dist = np.inf) -> np.ndarray:
        self.camera_img = self._query_rgb(state, max_dist)
        print(self.camera_img.shape)
        # self.camera_img = crop_to_aspect_ratio(self.camera_img, target_ratio=self.ratio)

        # self.im.set_array(np.moveaxis(self.camera_img, [0, 1, 2], [2, 0, 1]))
        self.im.set_array(self.camera_img)
        plt.pause(0.01)

        return self.camera_img
        # return cv2.resize(image, (self.height, self.width))

    def _query_rgb(self, state: VehicleState, max_dist = np.inf) -> np.ndarray:
        if self.finder is None or self.image_labels is None:
            raise ValueError("Self.finder is None or self.image_labels is None, please double check whether is camera is fully prepared or not")
        
        # rewrite the state in the form compatible with camera object
        x = state.x.x
        y = state.x.y
        psi = np.mod(state.e.psi, 2 * np.pi)

        # v_long = state.v.v_long
        # v_tran = state.v.v_tran
        # w = state.w.w_psi

        state = [[x, y, psi]]
        dists, indices = self.finder.kneighbors(state, n_neighbors=10000)
        dists, indices = dists.flatten(), indices.flatten()

        for i, d in enumerate(dists):
            index = indices[i]
            if d < max_dist and index in self.image_labels:
                print(f"current measured state is {state}, its distance to the closest grid point is {d}")
                image_token = self.image_labels[index]

                image_path = os.path.join(self.image_folder, image_token)
                    # image = np.moveaxis(
                image = np.array(Image.open(image_path))

                return image
                # if the image is in the CACHE memory, read it directly
                # if image_token in self.CACHE:
                #     return self.CACHE[image_token]
                
                # else:
                #     image_path = os.path.join(self.image_folder, image_token)
                #     # image = np.moveaxis(
                #     image = np.array(Image.open(image_path))
                #         # , [2, 0, 1], [0, 1, 2])
                #     self.update_CACHE(image_token, image)
                #     return image
        
        raise ValueError("Cannot find the image that corresponds to a state closest enough to the input state")
    
    def update_CACHE(self, image_token, image):

        add_probability = 0.1 # the probability with which the image is going to be added to the full CACHE
        # update the CACHE memory
        if len(self.CACHE) <= self.CACHE_SIZE:
            self.CACHE[image_token] = image
        else:
            #if the cache memory is full, randomly replace an existing element with the current image
            if np.random.uniform(low = 0.0, high = 1.0) < add_probability:
                self.CACHE.pop(random.choice(self.CACHE.keys()))
                self.CACHE[image_token] = image
    
    def query_depth(self, state: VehicleState) -> np.ndarray:
        raise NotImplementedError

    def query_depth_raw(self, state: VehicleState) -> np.ndarray:
        raise NotImplementedError

    def query_lidar(self, state: VehicleState) -> np.ndarray:
        raise NotImplementedError

def make_camera(height, width, data_path, intervals):
    """@return: a camera object
    @ intervals: how accurate the returned camera is [accuracy in x_coordinates(m), accuracy in y_coordinates(m), accuracy in psi(radius)]"""
    camera = BaseCamera(height, width, data_path)
    states, images, max_values, min_values = camera.data_prepare()
    camera.make_finder(max_values, min_values, intervals)
    camera.image_prepare(states, images)

    return camera

def camera_test():
    sample_size = 10
    samples = np.random.randint(low = 0, high = 1000, size = sample_size)
    fig, axis = plt.subplots(sample_size, 2)

    data_path = "E:/barc_data"
    load_path = "E:\camera_test2"
    width, height = 640, 480
    intervals = np.asarray([0.01, 0.01, np.pi / 6])
    camera = BaseCamera(load_mode = True, load_path = load_path)

    test_path = data_path + "/ParaDriveLocalComparison_Sep7_35.npz"
    data = np.load(test_path, allow_pickle = True)
    states, images = data["states"][ : , 3 : ], data["images"]

    for i, sample in enumerate(samples):
        state = states[sample]
        orig_image = images[sample]
        state = VehicleState(x =Position(x = state[0], y = state[1], z = 0.0), e = OrientationEuler(phi = 0.0, theta = 0.0, psi = state[2]))
        axis[i, 0].imshow(np.moveaxis(orig_image, [0, 1, 2], [2, 0, 1]))

        est_image = camera.query_rgb(state)
        axis[i, 1].imshow(np.moveaxis(est_image, [0, 1, 2], [2, 0, 1]))

    plt.show()

def main():
    data_path = os.path.expanduser("~/tmp/data")
    width, height = 640, 480
    # intervals = np.asarray([0.01, 0.01, np.pi / 6])
    # camera = make_camera(height, width, data_path, intervals)
    # state = VehicleState(x =Position(x = 1.0, y = 3.0, z = 0.0), e = OrientationEuler(phi = 0.0, theta = 0.0, psi = 0.5))
    # camera.query_rgb(state)
    # camera_test()
    camera = BaseCamera(height, width, load_mode = False)
    camera.train(data_path, save_path = os.path.expanduser("~/tmp/data/export"), model_name = "camera_test2")

if __name__ == "__main__":
    main()
