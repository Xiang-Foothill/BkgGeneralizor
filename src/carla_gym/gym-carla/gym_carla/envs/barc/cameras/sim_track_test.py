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

def main():
    client = carla.Client(host = 'localhost', port = 2000)
    client.set_timeout(10.0)
    track_name = 'sim_track1'
    xodr_path = Path(__file__).resolve().parents[1] / 'OpenDrive' / f"{track_name}.xodr"
    
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
    world = client.generate_opendrive_world(
                          data, carla.OpendriveGenerationParameters(
                              vertex_distance=vertex_distance,
                              max_road_length=max_road_length,
                              wall_height=wall_height,
                              additional_width=extra_width,
                              smooth_junctions=True,
                            enable_mesh_visibility=True))
    while True:
         world.tick()

if __name__ == '__main__':
    main()