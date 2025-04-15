import os
from abc import ABC, abstractmethod
import copy
from collections import deque
from pathlib import Path
from typing import Optional

from torchvision.transforms import transforms

import gym_carla
import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.carla_gym.controllers.barc_lmpc import LMPCWrapper
from src.carla_gym.controllers.barc_mpcc_conv import MPCCConvWrapper
from src.carla_gym.controllers.barc_pid import PIDWrapper
import models.feedforward
from models import safeAC, visionSafeAC
from models.base_model import BaseModel

from utils import data_util
from torch.utils.data import DataLoader
import utils.pytorch_util as ptu
from utils.logging.writer import MultiPurposeWriter

from loguru import logger
from labml import experiment

from il_trainer import IL_Trainer_CARLA, IL_Trainer_CARLA_SafeAC, IL_Trainer_CARLA_VisionSafeAC
DEFAULT_MODEL = "_L_track_barc_v1.2.3-lam1_230"

expert_mp = {
    'pid': PIDWrapper,
    'mpcc-conv': MPCCConvWrapper,
}

def test_load_model(model_name = DEFAULT_MODEL):

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_epochs', type=int, default=500)
    parser.add_argument('--initial_traj_len', type=int, default=1024)
    parser.add_argument('--n_training_per_epoch', type=int, default=1)
    parser.add_argument('--n_initial_training_epochs', type=int, default=5)
    parser.add_argument('--replay_buffer_maxsize', type=int, default=102_400)
    parser.add_argument('--expert', '-c', type=str, default='mpcc-conv',
                        choices=tuple(expert_mp.keys()))
    parser.add_argument('--observe', '-o', type=str, default='camera',
                        choices=('camera', 'state'))
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--comment', '-m', type=str, default='')
    parser.add_argument('--eps_len', type=int, default=1024)

    parser.add_argument('--town', type=str, default='L_track_barc')
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--no_saving', action='store_true')
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--beta', type=float, default=0.95)
    parser.add_argument('--beta_decay_freq', type=int, default=1)

    parser.add_argument('--evaluation', action='store_true')
    parser.add_argument('--continue', type=int, default=0)
    parser.add_argument('--freeze_weather', action='store_true')
    parser.add_argument('--fix_spawning', action='store_true')
    parser.add_argument('--pretrain_critic', action='store_true')

    parser.add_argument('--experimental', action='store_true')
    # parser.add_argument('--ntfy_freq', type=int, default=100)

    params = vars(parser.parse_args())

    if params['experimental']:
        params.update({
            # 'initial_traj_len': 128,
            'comment': '_'.join((params['comment'], 'experimental')),
            # 'ntfy_freq': -1,
        })

    print(params)

    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])

    comment = '_'.join((params['town'], params['comment']))

    params['comment'] = '_'.join((params['town'], params['comment'])) + ('_evaluation' if params['evaluation'] else '')

    t0, dt, dt_sim = 0., 0.1, 0.01

    carla_params = dict(
        track_name=params['town'],
        t0=t0, dt=dt, dt_sim=dt_sim,
        do_render=params['render'],
        max_n_laps=50,
        enable_camera=params['observe'] == 'camera',
        host=params['host'],
        port=params['port'],
    )

    with open(f"./config/{'visionSafeAC' if params['observe'] == 'camera' else 'safeAC'}.yaml", 'r') as f:
        config = yaml.safe_load(f)

    agent_params = config['model_hparams']

    trainer_cls = IL_Trainer_CARLA_VisionSafeAC if params['observe'] == 'camera' else IL_Trainer_CARLA_SafeAC

    trainer = trainer_cls(carla_params,
                          expert_cls=expert_mp[params['expert']],
                          replay_buffer_maxsize=params['replay_buffer_maxsize'],
                          eps_len=params['eps_len'],
                          initial_traj_len=params['initial_traj_len'],
                          do_relabel_with_expert=True,
                          n_training_per_epoch=params['n_training_per_epoch'],
                          comment=params['comment'],
                          no_saving=params['no_saving'],
                          starting_step=params['continue'],
                          eval_freq=params['eval_freq'],
                          batch_size=params['batch_size'],
                          n_initial_training_epochs=params['n_initial_training_epochs'],
                          beta=params['beta'],
                          # use_labml_tracker=not params['experimental'],
                          # ntfy_freq=params['ntfy_freq'],
                          beta_decay_freq=params['beta_decay_freq'],
                          pretrain_critic=params['pretrain_critic'],
                          **agent_params
                          )

    trainer = IL_Trainer_CARLA_VisionSafeAC
    trainer.agent.load(path=Path(__file__).resolve().parent / 'model_data' / 'significant_checkpoints',
                           name=model_name)

if __name__ == "__main__":
    test_load_model()