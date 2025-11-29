"""verify the correlation between the latent space conditional dicriminator loss and the agent's online performance in the target domain"""
from il_estimator import il_Estimator, generate_target_buffer
from il_CAD_BARC_trainer import BARC4_PID, BARC5_PID, PRETRAIN_BRIGHT1, PRETRAIN_BRIGHT2, PRETRAIN_BRIGHT3
from il_NR_trainer import DOMAIN1, DOMAIN2, DOMAIN5, DOMAIN6, DOMAIN7, DOMAIN8, DOMAIN9, DOMAIN10, DOMAIN11, DOMAIN12, DOMAIN13, DOMAIN14, IL_Trainer_CARLA_VisionNaiveRandomizationAC
import gym
import gym_carla
from loguru import logger
import yaml
from pathlib import Path
from models.visionSafeAC import VisionNaiveMultihead, VisionNaiveRandomization, VisionCompleteMultiHead, VisionAdversarialAdaptAC, VisionAdversarialActor, VisionConditionAdversarialAdaptAC, VisionNullAdaptAC, VisionProjConditionAdversarialAC
from utils.data_util import EfficientReplayBuffer
import numpy as np
from src.carla_gym.controllers.barc_pid import PIDWrapper
import utils.plot_util as pul
import torch
from mpclab_common.pytypes import VehicleState, VehicleActuation, VehiclePrediction, Position, ParametricPose, \
    BodyLinearVelocity, OrientationEuler, BodyAngularVelocity
import matplotlib.pyplot as plt
import carla

WEATHER_DIC = np.asarray([
    [0.0, 0.0, 90.0, 0.0],
    [80.0, 0.0, 90.0, 0.0],
    [0.0, 80.0, 90.0, 0.0],
    [0.0, 0.0, 3.0, 0.0],
    [0.0, 0.0, 5.0, 0.0],
    [0.0, 0.0, 0.5, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 80.0, 0.7, 0.0]
]
)

L_TRACK_BARC1 = '/Game/L_track_barc1/Maps/L_track_barc1/L_track_barc1' # same track shape as L_TRACK_BARC but with fences and trees
L_TRACK_BARC2 = '/Game/L_track_barc2/Maps/L_track_barc2/L_track_barc2'
L_TRACK_BARC4 = '/Game/L_track_barc4/Maps/L_track_barc4/L_track_barc4'
L_TRACK_BARC5 = '/Game/L_track_barc5/Maps/L_track_barc5/L_track_barc5'
L_TRACK_BARC6 = '/Game/L_track_barc6/Maps/L_track_barc6/L_track_barc6'

def visualize_domains(domain_list, pos=None):
    """
    For each domain in `domain_list`, reset the barc-v0 env with that
    map & weather, set env.sim_state to the same pose `pos`, grab the
    camera observation, and plot them in a row.

    If `pos` is None, a default pose is obtained from CARLA via the
    first spawn point of the current map.
    """
    t0, dt, dt_sim = 0., 0.1, 0.01
    carla_params = dict(
        track_name='L_track_barc',
        t0=t0, dt=dt, dt_sim=dt_sim,
        do_render=False,
        max_n_laps=50,
        enable_camera=True,
        host='localhost',
        port=2000,
    )

    env = gym.make('barc-v0', **carla_params)

    rgbs = []

    for domain in domain_list:
        cur_map   = domain["map_name"]
        weatherID = domain["weatherID"]

        # 1) Reset into this domain
        ob, info = env.reset(options={}, map_name=cur_map, weatherID=weatherID)

        # 2) Put the sim into the given pose
        env.sim_state = pos
        env.track_obj.global_to_local_typed(env.sim_state)
        
        # 3) Grab camera image from observation
        rgb = env.get_rgb()
        rgbs.append((rgb, cur_map, weatherID))

    # Plot all images in a row
    n = len(rgbs)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (img, map_name, weatherID) in zip(axes, rgbs):
        ax.imshow(img)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    env.close()
    
if __name__ == "__main__":
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    # Force-load your map
    world = client.load_world(L_TRACK_BARC6)
    weatherID = 2
    weather = carla.WeatherParameters(
                         cloudiness=WEATHER_DIC[weatherID][0],
                         precipitation=WEATHER_DIC[weatherID][1],
                         sun_altitude_angle=WEATHER_DIC[weatherID][2],
                         fog_density = WEATHER_DIC[weatherID][3])
    world.set_weather(weather)
    print("Current map:", world.get_map().name)
    spectator = world.get_spectator()
    
    spectator.set_transform(
    carla.Transform(
        carla.Location(x=-5.4, y=0.0, z=4.0),
        carla.Rotation(pitch=-30, yaw=0, roll=0)
    )
)

        