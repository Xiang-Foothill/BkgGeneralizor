import os
import sys

import numpy as np
from matplotlib import pyplot as plt
# import cv2

from mpclab_common.track import get_track
from mpclab_common.pytypes import VehicleState
import carla
from pathlib import Path
import torch
import random
from PIL import Image

K = 5 # the number of iteration used for mixing augmentation
BETA = 10 # the beta parameter used for specifying the beta distribution from which weights of mixing are sampled
FRACTAL_PATH = Path(__file__).parent / 'fractal_mixset.pth' # the path from which to load the mixing set, note that all the images in the mixset has shape of 64 by 64
ENV_PATH = Path(__file__).parent / 'env_mixset.pth'
MIX_MODE1 = "semantics"
MIX_MODE2 = "fractal"
SEMNATICS_SELECTED = [11]

"""there are two options to perform pixmixing, which can be specified by the MIX_MODE paramenter:
1. "semantics", a mixing function that applied semantic mask to bound the randomness of images to make sure 
the key features of images are not over blended.
2. "fractal", the mixing function that strictly follows the instructions of the paper with itertion of random beta mixing. The mixset will be a set
of fractal of images. This mode may sometimes lead to over-blending of images' key features.
"""

class mixCamera:
    def __init__(self, track_name, mix_mode = MIX_MODE2, semantics_selected = SEMNATICS_SELECTED, host='localhost', port=2000):
        print("""////////// Initiation of mix_camera starts ////////////""")
        if mix_mode == MIX_MODE1:
            self.mixset_path = ENV_PATH
            self.mix_func = semantics_mix
        if mix_mode == MIX_MODE2:
            self.mixset_path = FRACTAL_PATH
            self.mix_func = pixmix

        self.semantics_selected = semantics_selected
        self.mixset = torch.load(self.mixset_path)
        print( f"///// mixset successfully loaded with {self.mixset.shape[0]} images available /////")

        self.client = carla.Client(host, port)
        self.obs_size = 256
        self.dt = 0.1

        self.rgb_img = np.empty((self.obs_size, self.obs_size, 3), dtype=np.uint8)
        self.semantic_mask = np.empty((self.obs_size, self.obs_size, 3), dtype=np.uint8)
        self.mix_img = np.empty((self.obs_size, self.obs_size, 3), dtype=np.uint8)

        self.world = None
        self.camera_bp = None
        self.track_obj = get_track(track_name)

        # so far we are developing the test script that can be run without loading opendrive_map, get back this line later
        self.load_opendrive_map(track_name)
        # self.test_init()

        # Temporary: built-in rendering
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2)
        self.im1, self.im2 = self.ax1.imshow(self.rgb_img), self.ax2.imshow(self.mix_img)

        self.spawn_rgb_camera()
        self.spawn_semantic_camera()
        print("----- cameras successfully loaded -----")

    @property
    def height(self):
        return self.rgb_img.shape[0]
    
    @property
    def width(self):
        return self.rgb_img.shape[1]
    
    def test_init(self):
        self.world = self.client.get_world()# get rid of this line later when finishing devleoping
        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = True
        self.settings.fixed_delta_seconds = self.dt
        self.world.apply_settings(self.settings)

    def load_opendrive_map(self, track_name):
        xodr_path = Path(__file__).resolve().parents[1] / 'OpenDrive' / f"{track_name}.xodr"
        if not os.path.exists(xodr_path):
            raise ValueError("The file does not exist.")
            return
    
        with open(xodr_path, encoding='utf-8') as od_file:
            try:
                data = od_file.read()
            except OSError:
                print('file could not be read.')
                sys.exit()
        print('load opendrive map %r.' % os.path.basename(xodr_path))
        vertex_distance = 2.0  # in meters
        max_road_length = 0.1  # in meters
        wall_height = 0.2      # in meters
        extra_width = 0.1       # in meters
        self.world = self.client.generate_opendrive_world(
                        data, carla.OpendriveGenerationParameters(
                            vertex_distance=vertex_distance,
                            max_road_length=max_road_length,
                            wall_height=wall_height,
                            additional_width=extra_width,
                            smooth_junctions=True,
                            enable_mesh_visibility=True))
        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = True
        self.settings.fixed_delta_seconds = self.dt
        self.world.apply_settings(self.settings)

        # Next, try to spawn a camera at the origin. 
    def spawn_rgb_camera(self):
        self.rgb_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.rgb_bp.set_attribute('image_size_x', str(self.obs_size))
        self.rgb_bp.set_attribute('image_size_y', str(self.obs_size))
        self.rgb_bp.set_attribute('fov', '110')
        # Set the time in seconds between sensor captures
        self.rgb_bp.set_attribute('sensor_tick', f"{self.dt}")

        def get_rgb_img(data):
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.rgb_img = array

        self.camera_trans = carla.Transform(carla.Location(x=0, y=0, z=0.2))
        self.rgb_sensor = self.world.spawn_actor(self.rgb_bp, self.camera_trans)
        self.rgb_sensor.listen(get_rgb_img)
    
    def spawn_semantic_camera(self):
        self.semantic_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        self.semantic_bp.set_attribute('image_size_x', str(self.obs_size))
        self.semantic_bp.set_attribute('image_size_y', str(self.obs_size))
        self.semantic_bp.set_attribute('fov', '110')
        # Set the time in seconds between sensor captures
        self.semantic_bp.set_attribute('sensor_tick', f"{self.dt}")

        def mask_generator(data):
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]

            array = array.copy()
            array.flags.writeable = True
            array[:, :, 1] = array[:, :, 0]
            array[:, :, 2] = array[:, :, 0]
            semantic_tags = array

            self.semantic_mask = np.logical_or.reduce([semantic_tags == semantic for semantic in self.semantics_selected])

        self.camera_trans = carla.Transform(carla.Location(x=0, y=0, z=0.2))
        self.semantic_sensor = self.world.spawn_actor(self.semantic_bp, self.camera_trans)
        self.semantic_sensor.listen(mask_generator)
    
    def query_rgb(self, state):

        self.rgb_sensor.set_transform(carla.Transform(carla.Location(x=state.x.x, y=-state.x.y, z=0.2), 
                                                         carla.Rotation(yaw=-np.rad2deg(state.e.psi))))
        self.semantic_sensor.set_transform(carla.Transform(carla.Location(x=state.x.x, y=-state.x.y, z=1.0), 
                                                         carla.Rotation(yaw=-np.rad2deg(state.e.psi))))
        self.world.tick()
        self.buffer_64()

        # apply the mixing augmentatio below
        mixing_pic = self.mixset[np.random.randint(low = 0, high = self.mixset.shape[0])] # choose a random picture from the mixset
        self.mix_img = self.mix_func(x_orig = self.rgb_img, x_mix = mixing_pic, mask = self.semantic_mask)

        self.im1.set_array(self.rgb_img)
        self.im2.set_array(self.mix_img)

        plt.pause(1.0)
    
        return self.mix_img
    
    def buffer_64(self):
        def downsample(img):
                return np.array(Image.fromarray(img, mode='RGB').resize((64, 64)))
        self.rgb_img = downsample(self.rgb_img)
        self.semantic_mask = downsample(self.semantic_mask)

def semantics_mix(x_orig, x_mix, mask):
    """unlike the pixmix pipeline desribed in the paper in which x_org and x_mix are mixed based on randomnized iterations of
    beta distribution,
    semantics_mix mix two images in one shot with bounded additive blending.
    The gamma(weight) matrix for x_orig is caculated by:
    (1 - mask) * uniform(mask_bound, 1) + mask * uniform(0, 1 - mask_bound)"""

    mask_bound = 0.7
    gamma = (1 - mask) * np.random.uniform(mask_bound, 1) + mask * np.random.uniform(0, 1 - mask_bound)
    blended = x_orig * gamma + x_mix * (1 - gamma)
    blended = np.clip(blended.astype(np.uint8), None, 255)

    return blended

def pixmix(x_orig, x_mix, mask, k = K, beta = BETA):
    """This pixmix method strictly followed the mix-pipeline desribed in the paper.
    However, it turns out that it does not perform quite well. The key features of the graph are over-blended"""

    x_pixmix = random.choice([augment(x_orig), x_orig])

    for _ in range(random.choice(list(range(k)))):
        mix_image = random.choice([augment(x_orig), x_mix])
        mix_op = random.choice([mix_add, mix_mul])

        x_pixmix = mix_op(x_pixmix, mix_image, beta)
    
    return x_pixmix

def mix_add(x1, x2, beta):
    """additive blending between x1, and x2 with weights sampled from a beta distribution specified by beta"""
    alpha = 1.0

    weight2 = np.random.beta(alpha, beta)
    weight1 = 1 - weight2

    x_mix = (x1 * weight1 + x2 * weight2).astype(np.uint8)
    x_mix = np.clip(x_mix, None, 255)

    return x_mix

def mix_mul(x1, x2, beta):
    """multiplicative blending between x1, and x2 with weights sampled from a beta distribution specified by beta"""
    alpha = 1.0

    weight2 = np.random.beta(alpha, beta)
    weight1 = 1 - weight2

    x1, x2 = x1 / 255.0, x2 / 255.0
    x_mix = (x1 ** weight1) * (x2 ** weight2)
    x_mix = (x_mix * 255).astype(np.uint8)

    return x_mix

def augment(x):
    # list all the possible traditional image augmentation functions here
    aug_op = random.choice([
        random_cut,
        f_identity,
        posterize,
        solarize
    ])
    return aug_op(x)

"""the functions below are traditional augmentation functions"""
def f_identity(x):
    """return the original image without any augmentation"""
    return x

def posterize(x, bits=4):
    """Apply posterization by reducing color depth."""
    shift = 8 - bits  # Compute shift amount
    posterized = (x >> shift) << shift  # Reduce bit depth
    return posterized

def solarize(x, threshold = 128):
    solarized = np.copy(x)
    mask = x >= threshold
    solarized[mask] = 255 - solarized[mask]
    return solarized

def random_cut(x):
    cut = np.copy(x)
    h_low = np.random.randint(low = 0, high = x.shape[0] - 10)
    w_low = np.random.randint(low = 0, high = x.shape[1] - 10)
    h_high = np.random.randint(low = h_low + 1, high = x.shape[0])
    w_high = np.random.randint(low = w_low + 1, high = x.shape[1])
    
    cut[h_low : h_high, w_low : w_high, : ] = 0

    return cut


def carla_test():
    camera = mixCamera(track_name='L_track_barc')
    spawn_points = camera.world.get_map().get_spawn_points()

    for i, spawn_point in enumerate(spawn_points):
        state = VehicleState()
        state.x.x, state.x.y, state.e.psi = spawn_point.location.x, - spawn_point.location.y, np.random.uniform(low = 0.0, high = 2* np.pi)
        camera.query_rgb(state)
    
if __name__ == "__main__":
    # carla_test()
    carla_test()
