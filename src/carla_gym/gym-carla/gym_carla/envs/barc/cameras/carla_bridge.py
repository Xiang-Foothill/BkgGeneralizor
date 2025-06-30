import os
import sys

import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
# import cv2

from mpclab_common.track import get_track
from mpclab_common.pytypes import VehicleState

import carla
from pathlib import Path
import time

import pygame
import skimage
from gym_carla.envs.barc.cameras.distortor import original_image
import yaml
from gym_carla.envs.barc.barc_env import TRACK_NAME_SET

SEMNATICS_SELECTED = [11]
import gc
"""Listed below are available map names"""
L_TRACK_BARC = "L_track_barc" # the original map without any additional features
L_TRACK_BARC1 = '/Game/L_track_barc1/Maps/L_track_barc1/L_track_barc1' # same track shape as L_TRACK_BARC but with fences and trees
L_TRACK_BARC2 = '/Game/L_track_barc2/Maps/L_track_barc2/L_track_barc2'
L_TRACK_BARC4 = '/Game/L_track_barc4/Maps/L_track_barc4/L_track_barc4'
L_TRACK_BARC5 = '/Game/L_track_barc5/Maps/L_track_barc5/L_track_barc5'
L_TRACK_BARC6 = '/Game/L_track_barc6/Maps/L_track_barc6/L_track_barc6'
FOREST_SIM_TRACK1 = '/Game/forest_sim_track1/Maps/forsest_sim_track1/forsest_sim_track1'

DEBUG = True
WEATHER_DIC = np.asarray([
    [0.0, 0.0, 90.0, 0.0],
    [80.0, 0.0, 90.0, 0.0],
    [0.0, 80.0, 90.0, 0.0],
    [0.0, 0.0, 3.0, 0.0],
    [0.0, 0.0, 5.0, 0.0],
    [0.0, 0.0, 0.5, 0.0],
    [0.0, 0.0, 1.0, 0.0]
]
)

def rgb_to_display_surface(rgb, display_size):
    """
    Generate pygame surface given an rgb image uint8 matrix
    :param rgb: rgb image uint8 matrix
    :param display_size: display size
    :return: pygame surface
    """
    surface = pygame.Surface((display_size, display_size)).convert()
    display = skimage.transform.resize(rgb, (display_size, display_size))
    display = np.flip(display, axis=1)
    display = np.rot90(display, 1)
    pygame.surfarray.blit_array(surface, display)
    return surface


class CarlaConnector:
    def __init__(self, track_name, host='localhost', port=2000, weatherID = 0, map_name = L_TRACK_BARC):
        # self.client = carla.Client('localhost', 2000)
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        logger.debug(f"available maps: {self.client.get_available_maps()}")
        self.weatherID = weatherID
        self.obs_size = 224
        self.dt = 0.1

        self.rgb_img = np.empty((self.obs_size, self.obs_size, 3), dtype=np.uint8)
        self.semantic_mask = np.empty((self.obs_size, self.obs_size, 3), dtype=np.uint8)

        self.world = None
        self.camera_bp = None
        self.track_name = track_name
        self.track_obj = get_track(track_name)
        self.map_name = map_name

        self.load_map()
        self.set_world()
        self.shift_config()
        self.sensor_list = []

        self.spawn_camera()
        self.env_steps = 0
        
        # Temporary: built-in rendering
        if DEBUG:
            pygame.init()
            self.display = pygame.display.set_mode((224, 224), pygame.HWSURFACE | pygame.DOUBLEBUF)
            self.surface = pygame.Surface((self.obs_size, self.obs_size))
            self.clock = pygame.time.Clock()
            # self.fig, self.ax = plt.subplots()
            # self.im = self.ax.imshow(self.camera_img)
    
    @property
    def height(self):
        return self.rgb_img.shape[0]
    
    @property
    def width(self):
        return self.rgb_img.shape[1]
    
    def reset(self, map_name, weatherID):
        self.destroy_camera()
        del self.world
        gc.collect()

        self.weatherID = weatherID
        self.map_name = map_name

        self.load_map()
        self.set_world()
        self.shift_config()

        self.spawn_camera()
        self.env_steps = 0

    def load_map(self):
        if self.map_name == L_TRACK_BARC:
            self.load_opendrive_map()
        else:
            self.world = self.client.load_world(self.map_name)
    
    def shift_config(self):

        last_segment = self.map_name.strip().split('/')[-1]
        config_key = L_TRACK_BARC

        for track_name in TRACK_NAME_SET:
            if track_name in last_segment:
                config_key = track_name
                break
        
        """sometimes, due to the inconsistency between carla import and opendrive xdor format
        the origin point of the imported carla maps will shift, making the resulted coordinate system inconsistent with our mpc simulation"""
        with open("src/carla_gym/config/map_config.yaml", "r") as file:
            config_file = yaml.safe_load(file)

        self.xshift = config_file[config_key]["xshift"]
        self.yshift = config_file[config_key]["yshift"]
        
    def load_opendrive_map(self):
        xodr_path = Path(__file__).resolve().parents[1] / 'OpenDrive' / f"{self.track_name}.xodr"
        if not os.path.exists(xodr_path):
            raise ValueError(f"The file {xodr_path} does not exist.")
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
        wall_height = 0.01      # in meters
        extra_width = 0.1       # in meters
        self.world = self.client.generate_opendrive_world(
                          data, carla.OpendriveGenerationParameters(
                              vertex_distance=vertex_distance,
                              max_road_length=max_road_length,
                              wall_height=wall_height,
                              additional_width=extra_width,
                              smooth_junctions=True,
                            enable_mesh_visibility=True))
        

    def set_world(self):
        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = True
        self.settings.fixed_delta_seconds = self.dt
        self.world.apply_settings(self.settings)

        if self.weatherID != 0:
            weather = carla.WeatherParameters(
                         cloudiness=WEATHER_DIC[self.weatherID][0],
                         precipitation=WEATHER_DIC[self.weatherID][1],
                         sun_altitude_angle=WEATHER_DIC[self.weatherID][2],
                         fog_density = WEATHER_DIC[self.weatherID][3])

            self.world.set_weather(weather)

    def destroy_camera(self):
        for sensor in self.sensor_list:
            sensor.stop()
            sensor.destroy()
            gc.collect()
            # logger.info(f"Sensor {sensor.id} type: {sensor.type_id} is destroyed")
        self.sensor_list = []

    def spawn_camera(self):
        # Next, try to spawn a camera at the origin.
        self.spawn_rgb_camera()
        self.spawn_semantic_camera()
    
    def get_rgb_img(self, data):
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.rgb_img = array

    def spawn_rgb_camera(self):
        self.rgb_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.rgb_bp.set_attribute('image_size_x', str(self.obs_size))
        self.rgb_bp.set_attribute('image_size_y', str(self.obs_size))
        self.rgb_bp.set_attribute('fov', '110')
        # Set the time in seconds between sensor captures
        self.rgb_bp.set_attribute('sensor_tick', f"{self.dt}")

        self.camera_trans = carla.Transform(carla.Location(x=0, y=0, z=0.2))
        self.rgb_sensor = self.world.spawn_actor(self.rgb_bp, self.camera_trans)
        self.rgb_sensor.set_simulate_physics(False)
        self.rgb_sensor.listen(self.get_rgb_img)
        self.sensor_list.append(self.rgb_sensor)
        time.sleep(0.5)

    def mask_generator(self, data):
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]

            array = array.copy()
            array.flags.writeable = True
            array[:, :, 1] = array[:, :, 0]
            array[:, :, 2] = array[:, :, 0]
            semantic_tags = array

            self.semantic_mask = np.logical_or.reduce([semantic_tags == semantic for semantic in SEMNATICS_SELECTED])
    
    def spawn_semantic_camera(self):
        self.semantic_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        self.semantic_bp.set_attribute('image_size_x', str(self.obs_size))
        self.semantic_bp.set_attribute('image_size_y', str(self.obs_size))
        self.semantic_bp.set_attribute('fov', '110')
        # Set the time in seconds between sensor captures
        self.semantic_bp.set_attribute('sensor_tick', f"{self.dt}")

        self.camera_trans = carla.Transform(carla.Location(x=0, y=0.01, z=0.2))
        self.semantic_sensor = self.world.spawn_actor(self.semantic_bp, self.camera_trans)
        self.semantic_sensor.set_simulate_physics(False)
        self.semantic_sensor.listen(self.mask_generator)
        self.sensor_list.append(self.semantic_sensor)
        time.sleep(0.5)

    def query_rgb(self, state):
        self.env_steps += 1
        self.rgb_sensor.set_transform(carla.Transform(carla.Location(x=state.x.x + self.xshift, y=-state.x.y + self.yshift, z=0.2), 
                                                         carla.Rotation(yaw=-np.rad2deg(state.e.psi)))) # remember to add a system transfer function
        self.semantic_sensor.set_transform(carla.Transform(carla.Location(x=state.x.x, y=-state.x.y, z=0.2), 
                                                         carla.Rotation(yaw=-np.rad2deg(state.e.psi))))
        # attempt = 0
        # while True:
        #     try:
        for sensor in self.sensor_list:
            if not sensor.is_listening():
                logger.info("sensor is not listening")

        self.world.tick()
        if DEBUG:
            # surface = rgb_to_display_surface(self.camera_img, 256)
            display_image = self.rgb_img
            pygame.surfarray.blit_array(self.surface, display_image.swapaxes(0, 1))
            self.display.blit(self.surface, (0, 0))
            pygame.display.flip()
            self.clock.tick(60)
            # self.im.set_array(self.camera_img)
            # plt.pause(0.01)
            # self.fig.canvas.draw()
            # self.fig.canvas.flush_events()
            
        return self.rgb_img, self.semantic_mask
        # except RuntimeError as e:
        #     logger.error(e)
        #     logger.error("Waiting for CARLA to restart...")
        #     time.sleep(20)
        #     self.load_opendrive_map()
        #     self.spawn_camera(x=state.x.x, y=state.y.y, psi=state.e.psi)
        #     attempt += 1
        #     if attempt > 5:
        #         raise RuntimeError(e)

        # Built-in rendering
        # cv2.imshow('RGB Camera', self.camera_img[:, :, ::-1])
        # if cv2.waitKey(1) == ord('q'):
        #     exit(0)

    
    def test(self):
        # fig, ax = plt.subplots()
        # im = ax.imshow(self.camera_img)
        state = VehicleState()
        state.p.x_tran = 0.55
        
        while True:
            # im.set_array(self.camera_img)
            state.p.s = (state.p.s + 0.1) % self.track_obj.track_length

            self.track_obj.local_to_global_typed(state)
            print(self.query_rgb(state))
            # plt.pause(0.1)
            # fig.canvas.draw()
            # fig.canvas.flush_events()
    
    def get_test_data(self):
        "get the test data that will be sued for the distortor function"
        state = VehicleState()
        state.p.x_tran = 0.55
        res = []
        
        while len(res) <= 20:
            # im.set_array(self.camera_img)
            state.p.s = (state.p.s + 0.1) % self.track_obj.track_length

            self.track_obj.local_to_global_typed(state)
            image = self.query_rgb(state)
            
            if np.random.uniform(low = 0.0, high = 1.0) <= 0.3:
                res.append(image)

        res = np.asarray(res)
        np.save("test_images", res)

if __name__ == '__main__':
    connector = CarlaConnector(track_name='L_track_barc')
    connector.spawn_camera()
    connector.get_test_data()
