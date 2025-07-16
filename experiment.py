"""The experiment script for testing the domain adaptation algorithms"""
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

from loguru import logger
from labml import experiment

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

from il_CAD_trainer import IL_Trainer_CARLA_VisionAdversarialAdaptationAC
from il_PB_trainer import IL_Trainer_CARLA_Perfect_Baseline

def init_source_buffer(pretrain_model : str) -> data_util.EfficientReplayBuffer:
    """initialize the source domain data buffer"""
    randomnizor = linProgRandomnizer(final_percent=0.2, debug = False, mode = "constant", no_background = True)
    transform = {"camera": randomnizor.traditional_randomnize}

    #initialize the data buffer
    source_buffer: 'data_util.EfficientReplayBuffer' = None
    source_buffer = data_util.EfficientReplayBuffer(maxsize=20000,
                                    lazy_init=True,
                                    transform = transform)

    data_dir = Path(__file__).parent.parent / 'data'
    source_buffer.load(path = data_dir, name = pretrain_model)
    return source_buffer

def make_common_params(pretrain_model : str):
    """This method is used for hardcoding all the common parameters shared by the trainer classes"""
    # To save memory, all the trainers share the same source buffer and the carla gym
    source_buffer = init_source_buffer(pretrain_model)

    trainer_params = {
    'expert_cls': expert_mp['pid'],
    'replay_buffer_maxsize': 20000,
    'eps_len': 1024,
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
    'source_buffer' : source_buffer,
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
    trainer_params['env'] = env

    return trainer_params

def data_size_experiment(pretrain_model : str, title : str):
    """Test how much data is needed for domain adaptation. Plot max_traj_len of each of the model
    agains the size of the target domain data buffer size"""
    n_epochs = 8 # the maximum amount of epochs allowed for adaptation
    max_traj_lens = {}
    common_params = make_common_params(pretrain_model)
    tgt_domain_sizes = [2048, 1024, 512, 256, 128] #the sizes of the target domain buffer tahat is going to be verified

    trainer_clses = [(IL_Trainer_CARLA_Perfect_Baseline, None), 
                     (IL_Trainer_CARLA_VisionAdversarialAdaptationAC, {'discriminator_type' : 'cat_condition', 'sample_distribution' : 'naive_random'}),
                     (IL_Trainer_CARLA_VisionAdversarialAdaptationAC, {'discriminator_type' : 'cat_condition', 'sample_distribution' : 'first_4m_random'})]
    # trainer_clses = [(IL_Trainer_CARLA_Perfect_Baseline, None), 
    #                  (IL_Trainer_CARLA_VisionAdversarialAdaptationAC, {'discriminator_type' : 'cat_condition', 'sample_distribution' : 'naive_random'})]
    
    for tgt_domain_len in tgt_domain_sizes:
        for trainer_cls, predefined_params in trainer_clses:
            
            #specify the model params for the current class
            current_params = common_params.copy()
            current_params["target_domain_len"] = tgt_domain_len

            # add the predefine params
            if predefined_params is not None:
                for param_name in predefined_params:
                    current_params[param_name] = predefined_params[param_name]
            
            config_path = f"./config/{trainer_cls.__name__}.yaml"

            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            agent_params = config['model_hparams']
            trainer = trainer_cls(**current_params, **agent_params)
            max_traj_len = trainer.training_loop(n_epochs)

            if predefined_params is None:
                label = trainer_cls.__name__
            else:
                param_str = "_".join(str(v) for v in predefined_params.values())
                label = f"{trainer_cls.__name__}_{param_str}"

            if label not in max_traj_lens:
                max_traj_lens[label] = []
            max_traj_lens[label].append(max_traj_len)

    # Create the directory if it doesn't exist
    save_dir = os.path.join("graphs", "raw_data")
    os.makedirs(save_dir, exist_ok=True)

    # Prepare data
    save_path = os.path.join(save_dir, f"{title}.npz")
    np.savez_compressed(
        save_path,
        tgt_domain_sizes=np.array(tgt_domain_sizes),
        max_traj_lens=max_traj_lens  # keep as a dictionary
    )

    logger.info(f"Saved results (with dict) to {save_path}")

    visualize_data_size_experiment(title)

def data_size_experiment_with_variance(pretrain_model: str, title: str):
    """Test how much data is needed for domain adaptation.
    Plot max_traj_len of each model across different target buffer sizes.
    Each experiment is repeated 5 times with different seeds."""
    
    n_epochs = 6
    seeds = [0, 20, 35, 40, 80]

    num_seeds = len(seeds)

    max_traj_lens = {}
    tgt_domain_sizes = [2048, 1024, 512, 256, 213, 170, 128]
    # tgt_domain_sizes = [512, 256, 128]
    # tgt_domain_sizes = [1024]
    # trainer_clses = [
    #     (IL_Trainer_CARLA_Perfect_Baseline, None),
    #     (IL_Trainer_CARLA_VisionAdversarialAdaptationAC, {'discriminator_type': 'cat_condition', 'sample_distribution': 'naive_random'}),
    #     (IL_Trainer_CARLA_VisionAdversarialAdaptationAC, {'discriminator_type': 'cat_condition', 'sample_distribution': 'first_4m_random'}),
    #     (IL_Trainer_CARLA_VisionAdversarialAdaptationAC, {'discriminator_type': 'cat_condition', 'sample_distribution': 'middle_3m_random'})
    # ]

    # trainer_clses = [
    #     (IL_Trainer_CARLA_VisionAdversarialAdaptationAC, {'discriminator_type': 'cat_condition_reweight', 'sample_distribution': 'naive_random'}),
    #     (IL_Trainer_CARLA_VisionAdversarialAdaptationAC, {'discriminator_type': 'cat_condition_reweight', 'sample_distribution': 'first_4m_random'}),
    #     (IL_Trainer_CARLA_VisionAdversarialAdaptationAC, {'discriminator_type': 'cat_condition_reweight', 'sample_distribution': 'middle_3m_random'})
    # ]

    trainer_clses = [
        (IL_Trainer_CARLA_Perfect_Baseline, None),
        (IL_Trainer_CARLA_VisionAdversarialAdaptationAC, {'discriminator_type': 'cat_condition_pseudo', 'sample_distribution': 'naive_random'}),
        (IL_Trainer_CARLA_VisionAdversarialAdaptationAC, {'discriminator_type': 'cat_condition_pseudo', 'sample_distribution': 'first_4m_random'}),
        (IL_Trainer_CARLA_VisionAdversarialAdaptationAC, {'discriminator_type': 'cat_condition_pseudo', 'sample_distribution': 'middle_3m_random'})
    ]

    common_params = make_common_params(pretrain_model)

    for trainer_cls, predefined_params in trainer_clses:
        # Initialize storage: list of lists -> [len(tgt_domain_sizes)][num_seeds]
        results_matrix = []

        for tgt_domain_len in tgt_domain_sizes:
            seed_results = []

            for seed in seeds:
                current_params = common_params.copy()
                current_params["target_domain_len"] = tgt_domain_len

                # set the seed
                np.random.seed(seed)
                torch.manual_seed(seed)

                if predefined_params is not None:
                    current_params.update(predefined_params)

                config_path = f"./config/{trainer_cls.__name__}.yaml"
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)

                agent_params = config['model_hparams']
                trainer = trainer_cls(**current_params, **agent_params)
                max_traj_len = trainer.training_loop(n_epochs)
                seed_results.append(max_traj_len)

            results_matrix.append(seed_results)

        # Construct label
        if predefined_params is None:
            label = trainer_cls.__name__
        else:
            param_str = "_".join(str(v) for v in predefined_params.values())
            label = f"{trainer_cls.__name__}_{param_str}"

        # Store as a (len(tgt_domain_sizes), num_seeds) array
        max_traj_lens[label] = np.array(results_matrix)

    # Save data
    save_dir = os.path.join("graphs", "raw_data")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"{title}.npz")
    np.savez_compressed(
        save_path,
        tgt_domain_sizes=np.array(tgt_domain_sizes),
        max_traj_lens=max_traj_lens
    )

    logger.info(f"Saved results (with dict) to {save_path}")
    visualize_data_size_experiment_shadow(title)

def label_simplifier(label):
        """hard code method that makes the label cleaner when demonstrating"""
        if IL_Trainer_CARLA_Perfect_Baseline.__name__ in label:
            return "Perfect Baseline"
        elif IL_Trainer_CARLA_VisionAdversarialAdaptationAC.__name__ in label and 'cat_condition' in label and "naive_random" in label:
            return "TCADT(naive random distribution)"
        elif IL_Trainer_CARLA_VisionAdversarialAdaptationAC.__name__ in label and 'cat_condition' in label and "first_4m_random" in label:
            return "TCADT(first-4m heavy distribution)"
        elif IL_Trainer_CARLA_VisionAdversarialAdaptationAC.__name__ in label and 'cat_condition' in label and "middle_3m_random" in label:
            return "TCADT(middle-3m heavy distribution)"
        elif IL_Trainer_CARLA_VisionAdversarialAdaptationAC.__name__ in label and 'cat_condition_reweight' in label and "naive_random" in label:
            return "TCADTR(naive random distribution)"
        elif IL_Trainer_CARLA_VisionAdversarialAdaptationAC.__name__ in label and 'cat_condition_reweight' in label and "first_4m_random" in label:
            return "TCADTR(first-4m heavy distribution)"
        elif IL_Trainer_CARLA_VisionAdversarialAdaptationAC.__name__ in label and 'cat_condition_reweight' in label and "middle_3m_random" in label:
            return "TCADTR(middle-3m heavy distribution)"
        else:
            return label
        
def visualize_data_size_experiment(file_name):
    """
    Load and plot the data from a .npz file containing:
      - 'tgt_domain_sizes': a shared x-axis array
      - 'max_traj_lens': a dictionary mapping labels to y-axis arrays
    """

    save_dir = os.path.join("graphs", "raw_data")
    save_path = os.path.join(save_dir, f"{file_name}.npz")

    # Load data
    data = np.load(save_path, allow_pickle=True)
    tgt_domain_sizes = data['tgt_domain_sizes']
    max_traj_lens = data['max_traj_lens'].item()  # convert object to dict

    # Plot
    plt.figure(figsize=(8, 5))
    for label, traj_lens in max_traj_lens.items():
        clean_label = label_simplifier(label)
        plt.plot(tgt_domain_sizes, traj_lens, marker='o', label=clean_label)

    plt.xlabel('Target Domain Buffer Size')
    plt.ylabel('Maximum Trajectory Length')
    plt.title('Data Size vs. Adaptation Performance')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_data_size_experiment_with_variance(file_name):
    """
    Load and plot the data from a .npz file containing:
      - 'tgt_domain_sizes': a shared x-axis array
      - 'max_traj_lens': a dictionary mapping labels to 2D arrays [n_sizes, n_seeds]
    Plot the mean trajectory length with error bars showing standard deviation.
    """

    # Load file
    save_dir = os.path.join("graphs", "raw_data")
    save_path = os.path.join(save_dir, f"{file_name}.npz")
    data = np.load(save_path, allow_pickle=True)
    
    tgt_domain_sizes = data['tgt_domain_sizes']
    max_traj_lens = data['max_traj_lens'].item()  # dict of [n_sizes, n_seeds] arrays

    # Plot
    plt.figure(figsize=(8, 5))
    for label, traj_lens_matrix in max_traj_lens.items():
        traj_lens_matrix = np.array(traj_lens_matrix)
        means = traj_lens_matrix.mean(axis=1)
        stds = traj_lens_matrix.std(axis=1)
        clean_label = label_simplifier(label)
        plt.errorbar(tgt_domain_sizes, means, yerr=stds, fmt='-o', capsize=4, label=clean_label)

    plt.xlabel('Target Domain Buffer Size')
    plt.ylabel('Maximum Trajectory Length')
    plt.title('Data Size vs. Adaptation Performance')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_data_size_experiment_shadow(file_name):
    """
    Load and plot the data from a .npz file containing:
      - 'tgt_domain_sizes': a shared x-axis array
      - 'max_traj_lens': a dictionary mapping labels to 2D arrays [n_sizes, n_seeds]
    Plot the mean trajectory length with shaded variance region (mean ± std).
    """
    # Load file
    save_dir = os.path.join("graphs", "raw_data")
    save_path = os.path.join(save_dir, f"{file_name}.npz")
    data = np.load(save_path, allow_pickle=True)
    
    tgt_domain_sizes = data['tgt_domain_sizes']
    max_traj_lens = data['max_traj_lens'].item()  # dict of [n_sizes, n_seeds] arrays

    # Plot
    plt.figure(figsize=(8, 5))
    for label, traj_lens_matrix in max_traj_lens.items():
        traj_lens_matrix = np.array(traj_lens_matrix)
        means = traj_lens_matrix.mean(axis=1)
        stds = traj_lens_matrix.std(axis=1)

        clean_label = label_simplifier(label)

        # Decide line style by hardcoding
        if IL_Trainer_CARLA_VisionAdversarialAdaptationAC.__name__ not in label:
            line_style = ':'  # dotted
        else:
            line_style = '-'  # solid
        
        # Plot mean with 'x' marker and line
        plt.plot(tgt_domain_sizes, means, label=clean_label, marker='o', linestyle=line_style, linewidth=2)

        # Plot shaded region for standard deviation
        plt.fill_between(tgt_domain_sizes, means - stds, means + stds, alpha=0.2)

    plt.xlabel('Target Domain Buffer Size')
    plt.ylabel('Maximum Trajectory Length')
    plt.title('Data Size vs. Adaptation Performance')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    pretrain_model = PRETRAIN_BRIGHT3
    title = "data_size_pseudo_only_state_exp1"
    # data_size_experiment_with_variance(pretrain_model, title)
    visualize_data_size_experiment_shadow(title)