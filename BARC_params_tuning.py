"""Grid searching over the params space of CAD il_trainer to find the optimum parameters"""

import os
from abc import ABC, abstractmethod
import copy
from collections import deque
from pathlib import Path
from typing import Optional

import gym
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
from models.visionSafeAC import VisionNaiveMultihead, VisionNaiveRandomization, VisionCompleteMultiHead, VisionAdversarialAdaptAC, VisionAdversarialActor, VisionConditionAdversarialAdaptAC, VisionNullAdaptAC, VisionProjConditionAdversarialAC
from models.base_model import BaseModel
from domain_randomnization.randomnizor import BkgRandomnizer, linProgRandomnizer, ContrastRandomnizer
from models.adaptAC import WassersteinAdversarialAdaptAC, WassersteinConditionAdversarialAdaptAC, VisionConditionAdversarialReweightAdaptAC

from utils import data_util
from torch.utils.data import DataLoader
import utils.pytorch_util as ptu
from utils.logging.writer import MultiPurposeWriter
from il_trainer import IL_Trainer_CARLA_VisionSafeAC
import pickle
from sklearn.decomposition import PCA
from torch.utils.tensorboard import SummaryWriter

from loguru import logger
from labml import experiment
from itertools import product
import warnings
from copy import deepcopy
from typing import Dict, Iterable, List, Any

EVAL_MODEL1 = "L_track_barc_v1.2.3-lam1_230"
EVAL_MODEL2 = "L_track_barc_v1.4.2-lam1_57"
EVAL_MODEL3 = "L_track_barc_v1.2.3-lam1"

"""Listed below are available map names"""
L_TRACK_BARC = "L_track_barc" # the original map without any additional features

L_TRACK_BARC1 = '/Game/L_track_barc1/Maps/L_track_barc1/L_track_barc1' # same track shape as L_TRACK_BARC but with fences and trees
L_TRACK_BARC2 = '/Game/L_track_barc2/Maps/L_track_barc2/L_track_barc2'
L_TRACK_BARC4 = '/Game/L_track_barc4/Maps/L_track_barc4/L_track_barc4'
L_TRACK_BARC5 = '/Game/L_track_barc5/Maps/L_track_barc5/L_track_barc5'
L_TRACK_BARC6 = '/Game/L_track_barc6/Maps/L_track_barc6/L_track_barc6'

""""""
PRETRAIN_ENCODER1 = 'L_track_barc_NR_pretrain1'
PRETRAIN_ENCODER2 = 'L_track_barc_pretrain2'
PRETRAIN_ENCODER3 = 'L_track_barc_pretrain3' # this pretrained encoder uses the domain list of [DOMAIN2, DOMAIN4, DOMAIN8], the model output: visual encoder - 512, velocity encoder - 16
PRETRAIN_ENCODER4 = 'L_track_barc_pretrain4' # this pretrained encoder uses the domain list of [DOMAIN2, DOMAIN4, DOMAIN5, DOMAIN6, DOMAIN8], the model output: visual encoder - 512, velocity encoder - 16
PRETRAIN_ENCODER5 = 'L_track_barc_pretrain5' # domain list: [DOMAIN2, DOMAIN4, DOMAIN8]; eval domain list: [DOMAIN5, DOMAIN7]; the model output: visual encoder - 512, velocity encoder - 16
PRETRAIN_ENCODER6 = 'L_track_barc_pretrain6' # domain list: [DOMAIN2, DOMAIN4, DOMAIN8]; eval domain list: [DOMAIN5, DOMAIN7]; the model output: visual encoder - 512, velocity encoder - 16 [unlike pretrain_encoder5 this encoder is always saved at a global step of 5]
PRETRAIN_ENCODER7 = 'L_track_barc_pretrain7' # domain list: [DOMAIN2, DOMAIN4, DOMAIN8]; eval domain list: [DOMAIN5, DOMAIN7]; the model output: visual encoder - 256, velocity encoder - 16 [saved at a global step of 3]
PRETRAIN_ENCODER8 = 'L_track_barc_pretrain8' # domain list: [DOMAIN2, DOMAIN4, DOMAIN8]; eval domain list: [DOMAIN5, DOMAIN7]; the model output: visual encoder - 256, velocity encoder - 16 [saved at a global step of 6]
PRETRAIN_ENCODER9 = 'L_track_barc_many_domains_test2' # domain list: [DOMAIN2, DOMAIN8, DOMAIN9, DOMAIN4, DOMAIN1]; eval domain list: [DOMAIN5, DOMAIN7], visual encoder output: 512

PRETRAIN_LIGHTS = 'L_track_barc_light_domains' #domain list: [DOMAIN4, DOMAIN7, DOMAIN8]; visual_encoder output size: 512
PRETRAIN_LIGHTS2 = 'L_track_barc_light_domains2' #domain list: [DOMAIN4, DOMAIN7, DOMAIN8]; visual_encoder output size: 512
PRETRAIN_BRIGHT1 = 'L_track_barc_pretrain_bright1' # domain list: DOMAIN7, DOMAIN8; visual_encoder output size: 512
PRETRAIN_BRIGHT2 = 'L_track_barc_pretrain_bright2' #domain list: DOMAIN7, DOMAIN6, DOMAIN4; visual encoder output size: 512; Trained for 17 epochs
PRETRAIN_BRIGHT3 = 'L_track_barc_pretrain_bright3' #domain list: DOMAIN7, DOMAIN6, DOMAIN4; visual encoder output size: 512; Trained for 16 epochs

expert_mp = {
    'pid': PIDWrapper,
    'mpcc-conv': MPCCConvWrapper,
}
from il_NR_trainer import DOMAIN1, DOMAIN2, DOMAIN4, DOMAIN5, DOMAIN6, DOMAIN7, DOMAIN8, DOMAIN9, DOMAIN11, DOMAIN12, DOMAIN14, FULL_EVALUATION_LIST
from il_CAD_BARC_trainer import BARC0, BARC1, BARC2, BARC3
from il_CAD_BARC_trainer import IL_Trainer_CARLA_BARC_VisionAdversarialAdaptationAC

def make_common_params():
    """This method is used for hardcoding all the common parameters shared by the trainer classes"""
    # To save memory, all the trainers share the same source buffer and the carla gym

    trainer_params = {
    'expert_cls': expert_mp['mpcc-conv'],
    'discriminator_type' : 'cat_condition',
    'replay_buffer_maxsize': 20000,
    'eps_len': 128,
    'initial_traj_len': 3072,
    'do_relabel_with_expert': True,
    'n_training_per_epoch': 1,
    'no_saving': False,
    'starting_step': 0,
    'eval_freq': 1,
    'batch_size': 64,
    'n_initial_training_epochs': 1,
    'beta': 0.90,
    'beta_decay_freq': 1,
    'pretrain_critic': False,
    'save_profile': False,
    'to_reload': False,
    'latent': False,
    'mid_freeze': float('inf'),
    'to_PCA': False,
    'source_buffer' : None, # Note that each time we restart the experiment, the source buffer needs to be reloaded from scratch
}
    #initialize the carla env

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

    trainer_params['carla_params'] = carla_params
    env = gym.make('barc-v0', **carla_params)
    trainer_params['env'] = env # the gym environment is shared across domains

    return trainer_params

def make_agent_params(hyper_params: Dict[str, Iterable]) -> List[Dict[str, Any]]:
    """
    Build all combinations of hyper_params and union each with shared_params.
    If a key exists in both shared_params and the combo, warn and keep the combo's value.
    """
    shared_params = {
        "encoder_output_dim": 512,
        "dis_info_mode": "gps_full",
        "weight_decay": 1e-4,
    }

    # Edge case: no hyper-params provided
    if not hyper_params:
        return [deepcopy(shared_params)]

    # Ensure values are iterable (lists/tuples). Strings are not split.
    keys = list(hyper_params.keys())
    values_lists = []
    for k in keys:
        v = hyper_params[k]
        if isinstance(v, (str, bytes)):
            values_lists.append([v])
        else:
            try:
                values_lists.append(list(v))
            except TypeError:
                values_lists.append([v])

        # If any list is empty, no combinations exist
        if len(values_lists[-1]) == 0:
            return []

    combos = []
    for values in product(*values_lists):
        combo = dict(zip(keys, values))
        merged = deepcopy(shared_params)

        # Handle overlaps: warn and override with combo value
        for k, v in combo.items():
            if k in merged:
                warnings.warn(
                    f"Key '{k}' exists in both shared_params and hyper_params; "
                    f"using hyper_params value ({v}) over shared_params value ({merged[k]})."
                )
            merged[k] = v

        combos.append(merged)

    return combos

def make_run_name(hp):
    # keep it short; sanitize floats
    def f(x): return f"{x:.2g}" if isinstance(x, float) else str(x)
    return f"lrD{f(hp['lr_discriminator'])}_lrA{f(hp['lr_actor'])}_adv{f(hp['adv_factor'])}"

def grid_search(hyper_params : dict, n_epochs : int):
    """grid searching throughout the hyper_params dictionary.
    @hyper_params: a dictionary containing all the possible hyper parameters to be verified. The key is the paramter's name, and the
    corresponding value is the a list containing all possible paramters"""
    common_params = make_common_params()
    agent_params_combs = make_agent_params(hyper_params=hyper_params)
    
    for agent_params in agent_params_combs:
        comment = make_run_name(agent_params)
        common_params['comment'] = comment
        logger.info(f"------------- Verify the hyper params: {comment} ------------- ")
        trainer = IL_Trainer_CARLA_BARC_VisionAdversarialAdaptationAC(**common_params, **agent_params)
        trainer.training_loop(n_epochs)
    
def test_make_hyper_params(hyper_params):
    agent_param_combs = make_agent_params(hyper_params)
    for agent_param in agent_param_combs:
        logger.debug(make_run_name(agent_param))

if __name__ == '__main__':
    n_epochs = 15
    hyper_params = {
        'adv_factor' : [0.8, 0.25, 0.5],
        'lr_discriminator' : [5e-4, 1e-4, 1e-5],
        'lr_actor': [5e-4, 1e-4, 5e-5, 1e-5, 5e-6]
    }
    grid_search(hyper_params, n_epochs = n_epochs)