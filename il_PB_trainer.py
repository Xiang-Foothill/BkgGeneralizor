"""The implementation of Perfect Baseline:
This trainer assumes that the agent has full access to the target domain system:
1. expert labels are accessible
2. rolling trajectories are available
The newly collected data from the target domain are directly mixed with the data from the source domain"""

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
from models import safeAC, visionSafeAC
from models.base_model import BaseModel
from domain_randomnization.randomnizor import BkgRandomnizer, linProgRandomnizer, ContrastRandomnizer

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
FOREST_SIM_TRACK1 = '/Game/forest_sim_track1/Maps/forsest_sim_track1/forsest_sim_track1'

expert_mp = {
    'pid': PIDWrapper,
    'mpcc-conv': MPCCConvWrapper,
}

DOMAIN1 = {"name": "lawn with fence", "map_name" : L_TRACK_BARC1, "weatherID" : 0}
DOMAIN2 = {"name": "lawn when sunset", "map_name": L_TRACK_BARC2, "weatherID" : 3}
DOMAIN4 = {"name": "traffic corns", "map_name": L_TRACK_BARC4, "weatherID" : 0}
DOMAIN5 = {"name": "forests before sunset", "map_name": L_TRACK_BARC5, "weatherID" : 4}
DOMAIN6 = {"name": "bus stops before sunset", "map_name": L_TRACK_BARC6, "weatherID" : 4}

# DOMAIN7 and DOMAIN8 are very similar to each other
DOMAIN7 = {"name": "lawn at the noon", "map_name": L_TRACK_BARC2, "weatherID" : 0}
DOMAIN8 = {"name": "bus stops at the noon", "map_name": L_TRACK_BARC6, "weatherID" : 0}
DOMAIN9 = {"name": "fences when the sky is really dark", "map_name": L_TRACK_BARC1, "weatherID": 5}
DOMAIN10 = {"name": "bus stops when the sky is really dark", "map_name": L_TRACK_BARC6, "weatherID" : 6}
DOMAIN11 = {"name": "traffic corns when the sky is really dark", "map_name": L_TRACK_BARC4, "weatherID" : 6}
DOMAIN12 = {"name": "lawn when sky is really dark", "map_name": L_TRACK_BARC2, "weatherID" : 5}
DOMAIN13 = {"name": "forest_sim_track1_noon", "map_name": FOREST_SIM_TRACK1, "weatherID" : 0}

FULL_EVALUATION_LIST = [DOMAIN1, DOMAIN2, DOMAIN4, DOMAIN5, DOMAIN6, DOMAIN7, DOMAIN8, DOMAIN10, DOMAIN11, DOMAIN12]

PRETRAIN_LIGHTS = 'L_track_barc_light_domains' #domain list: [DOMAIN4, DOMAIN7, DOMAIN8]; visual_encoder output size: 512
PRETRAIN_LIGHTS2 = 'L_track_barc_light_domains2' #domain list: [DOMAIN4, DOMAIN7, DOMAIN8]; visual_encoder output size: 512

class IL_Trainer_CARLA_Perfect_Baseline(IL_Trainer_CARLA_VisionSafeAC):

    def __init__(self, carla_params,
                       expert_cls,
                       augment_percent=0.6,
                       initial_traj_len=1024,
                       eps_len=1024,
                       replay_buffer_maxsize=65536,
                       do_relabel_with_expert=True,
                       n_training_per_epoch=1,
                       comment='',
                       no_saving=False,
                       starting_step=0,
                       eval_freq=1,
                       batch_size=1,
                       n_initial_training_epochs=5,
                       beta=0.25,
                       pretrain_critic=False,
                       beta_decay_freq=5,
                       save_profile = True,
                       to_reload = False,
                       latent = False,
                       mid_freeze = np.inf,
                       to_PCA = False,
                       save_data = True,
                       **agent_params):
        """

        @param expert: A wrapper designed for CARLA. Must have a step function that accepts a single frame of
        observation from gym_carla, and returns a 1d numpy array of expert action.
        The expert must be deterministic.
        @param carla_params:
        @param initial_traj_len:
        @param replay_buffer_maxsize:
        @param do_relabel_with_expert:
        @param n_training_per_epoch:
        @param comment:
        @param no_saving:
        @param starting_step:
        @param eval_freq:
        @param batch_size:
        @param n_initial_training_epochs:
        @param agent_params:
        """

        self.pretrain_encoder_path = PRETRAIN_LIGHTS2
        self.target_domains = [DOMAIN12]

        self.to_PCA = to_PCA
        self.pretrain_saved = False
        self.cur_epoch = 0
        self.best_avg_lap_time = np.inf
        self.beta_decay_freq = beta_decay_freq
        self.eval_freq = eval_freq
        self.comment = comment
        self.initial_traj_len = initial_traj_len
        self.do_relabel_with_expert = do_relabel_with_expert
        self.n_training_per_epoch = n_training_per_epoch
        self.batch_size = batch_size
        self.no_saving = no_saving
        self.n_initial_training_epochs = n_initial_training_epochs
        self.init_beta = 0.2
        self.beta = beta
        # self.use_labml_tracker = use_labml_tracker
        self.agent_params = agent_params
        self.mid_freeze = mid_freeze

        self.n_eval_success, self.n_eval_total = 0, 0
        self.eval_rewards_last10 = deque(maxlen=10)
        self.visualize_freq = 5 # the frequency of visualizing latent vectors in terms epoc num

        self.update_carla_params(carla_params)
        self.env = gym.make('barc-v0', **carla_params)
        self.eps_len = min(replay_buffer_maxsize, eps_len)

        self.expert_cls = expert_cls
        self.carla_params = carla_params

        self.expert = self.expert_cls(dt=self.carla_params['dt'], t0=self.carla_params['t0'], track_obj=self.env.get_track())
        self.agent: 'BaseModel' = None
        self.save_data = save_data

        # check whether the user sets latent mode and pretrain mode at the same time
        if latent:
            save_profile = False
            to_reload = False # In the latent visualizaiton mode, it is default not allowed to save or reload any thing
            plt.figure() # initialize the plot space of latent space

        self.initialize_agent(comment=comment, latent = latent, **agent_params)
        self.latent = latent

        self.save_profile = save_profile

        # an extra parameter
        self.randomnizor = linProgRandomnizer(final_percent=0.2, debug = False, mode = "constant") # set debug to false to speed up rendering
        transform = {"camera": self.randomnizor.traditional_randomnize}

        self.replay_buffer: 'data_util.EfficientReplayBuffer' = None
        self.replay_buffer = data_util.EfficientReplayBuffer(maxsize=replay_buffer_maxsize,
                                                               lazy_init=True,
                                                               transform = transform
                                                               )

        self.writer: 'MultiPurposeWriter' = MultiPurposeWriter(model_name=self.agent.model_name,
                                                               log_dir=f"logs/{self.agent.model_name}_{comment or ''}",
                                                               comment=comment or '',
                                                               print_method=logger.info,
                                                               # use_labml_tracker=use_labml_tracker,
                                                               # ntfy_freq=ntfy_freq,
                                                               )

        # Load previous model weights and replay buffer.
        self.starting_step = starting_step
        self.agent.to(ptu.device)
        if starting_step > 0:
            self.agent.load()
            self.replay_buffer.load()
        if pretrain_critic:
            self.pretrain_critic()
        
        self.domain_list = [DOMAIN7, DOMAIN8] # the training-time available domains
        self.eval_domain_list = [] # the additional domains other than trianing-time available domains used for evaluation only

        # the list used to store evaluation result
        self.evaluation_list = {}
        for domain in (self.domain_list + self.eval_domain_list):
            self.evaluation_list[domain["name"]] = {}
        
        # Reload the agent parameters from the pretrain agent
        self.agent.load(path=Path(__file__).resolve().parent / 'model_data' / 'pretrained_agents',
                    name=self.pretrain_encoder_path)
        
        profile_path = Path(__file__).parent.parent / 'training_profiles' / f"{self.pretrain_encoder_path}_training_profile.pkl"
            
        data_dir = Path(__file__).parent.parent / 'data'
        self.replay_buffer.load(path = data_dir, name = self.pretrain_encoder_path) # reload the data buffer from the pretrain agent
        
        if self.starting_step >= self.mid_freeze:
            self.agent.freeze_encoders()
    
    def label_domain(self, ob, domain):
        """add the field 'domain_v', the domain probability vector to the ob
         @ob: the observation dictionary returned from the carla-gym environment
         @domain: the domain we used to so far"""
        domain_indicator = np.asarray([0.]) # set the indicator to zero if the domaindoes not belong the training_time available domains
        if domain in self.domain_list:
            domain_indicator = np.asarray([1.])
        ob['domain_indicator'] = domain_indicator
        return ob
    
    def initialize_agent(self, comment, latent, **kwargs):
        logger.debug(f"{kwargs}")
        if latent:
            self.agent =visionSafeAC.VisionNaiveRandomization_Visualization(**kwargs)
        else:
            self.agent = visionSafeAC.VisionNaiveRandomization(**kwargs)
    
    def initialize_replay_buffer():
        return None
    
    def train_module(self, module, global_step):
        logger.info(f"Training {module.__class__.__name__}...")
        info = module.fit(train_dataset=self.replay_buffer,
                          n_epochs=self.n_training_per_epoch if global_step > 0 else self.n_initial_training_epochs,
                          global_step=global_step)
        for mode, values in info.items():
            self.writer.do_logging(values, global_step=global_step, mode=mode)
    
    def sample_trajectory(self, domain, beta: float, pbar: Optional['tqdm'] = None,
                          max_traj_len=np.inf,
                          PATIENCE=2, TRUNCATE=np.inf):
        """

        @param beta:
        @param pbar:
        @param max_traj_len:
        @param PATIENCE: Maximum allowed consecutive expert fails before truncating the trajectory.
        @param TRUNCATE: Number of examples to remove from the replay buffer if the trajectory is truncated.
        @return:
        """
        cur_map = domain["map_name"]
        weatherID = domain["weatherID"]

        ob, info = self.env.reset(options={}, map_name = cur_map, weatherID = weatherID)

        self.agent.reset()
        self.agent.eval()
        self.expert.reset(options=info)
        # self.replay_buffer.clear_buffer() # note that the replay buffer is EfficientReplayBuffer instead of EffieicentReplayBufferPN
        terminated, truncated = False, False
        traj_len = 0
        fail_counter = 0

        while traj_len < max_traj_len:

            self.label_domain(ob, domain)
            ac = self.agent.get_action(*self.agent.parse_carla_obs(ob, info))
            # expert_ac, expert_info = self.expert.step(state=info['vehicle_state'],
            #                                           terminated=terminated,
            #                                           lap_no=info['lap_no'])
            expert_ac, expert_info = self.expert.step(**ob, **info)
            expert_ac = np.clip(expert_ac, self.env.action_space.low, self.env.action_space.high)
            closed_loop_action = beta * expert_ac + (1 - beta) * ac

            # try:
            if expert_info['success']:
                # action = expert_ac if np.random.rand() <= beta else ac
                # closed_loop_action = beta * expert_ac + (1 - beta) * ac
                next_ob, rew, terminated, truncated, info = self.env.step(closed_loop_action)
                # logger.debug(f"Action: {ac}, Expert action: {expert_ac}, v_long: {ob['state'][0]}")
                # self.add_frame(ob=ob, ac_agent=ac, ac_expert=expert_ac, rew=rew, terminated=terminated,
                #                truncated=truncated, info=info, next_ob=next_ob)
                self.replay_buffer.add_frame(ob, rew, terminated, truncated, info,
                                             action=expert_ac.astype(np.float32),
                                             closed_loop_action=closed_loop_action.astype(np.float32),
                                             next_state=next_ob['state'])
                fail_counter = 0

            else:
                logger.warning(f"Expert solved inaccurate with code {expert_info.get('status', 'unknown')}.")
                next_ob, rew, terminated, truncated, info = self.env.step(closed_loop_action)
                fail_counter += 1
                self.replay_buffer.add_frame(ob, rew, terminated, truncated, info,
                                                   action=expert_ac.astype(np.float32),
                                                   closed_loop_action=closed_loop_action.astype(np.float32),
                                                   next_state=next_ob['state'])
                if fail_counter >= PATIENCE:
                    truncated = True

            traj_len += 1
            ob = next_ob

            if pbar is not None:
                pbar.update(1)
            
            if truncated: # reset the environment if the vehicle is truncated
                logger.info(f"the vehicle is truncated, now respawning")
                ob, info = self.env.reset(options={'controller': self.expert})
                self.agent.reset()
                self.agent.eval()
                self.expert.reset(options=info)
                terminated, truncated = False, False

        return traj_len
    
    def sample_trajectory_with_future(self, domain, beta: float, pbar: Optional['tqdm'] = None,
                          max_traj_len=np.inf,
                          PATIENCE=2, TRUNCATE=np.inf):
        
        cur_map = domain["map_name"]
        weatherID = domain["weatherID"]

        ob, info = self.env.reset(options={}, map_name=cur_map, weatherID=weatherID)

        self.agent.reset()
        self.agent.eval()
        self.expert.reset(options=info)

        terminated, truncated = False, False
        traj_len = 0
        fail_counter = 0
        trajectory = []

        while traj_len < max_traj_len:
            self.label_domain(ob, domain= domain)
            ac = self.agent.get_action(*self.agent.parse_carla_obs(ob, info))
            expert_ac, expert_info = self.expert.step(**ob, **info)
            expert_ac = np.clip(expert_ac, self.env.action_space.low, self.env.action_space.high)
            closed_loop_action = beta * expert_ac + (1 - beta) * ac

            if expert_info['success']:
                next_ob, rew, terminated, truncated, info = self.env.step(closed_loop_action)
                fail_counter = 0
            else:
                logger.warning(f"Expert solved inaccurate with code {expert_info.get('status', 'unknown')}.")
                next_ob, rew, terminated, truncated, info = self.env.step(closed_loop_action)
                fail_counter += 1
                if fail_counter >= PATIENCE:
                    truncated = True

            # Store all info necessary for add_frame in a dict
            frame = {
                'ob': ob,
                'rew': rew,
                'terminated': terminated,
                'truncated': truncated,
                'info': info,
                'action': expert_ac.astype(np.float32),
                'closed_loop_action': closed_loop_action.astype(np.float32),
                'next_state': next_ob['state'],
            }
            trajectory.append(frame)

            traj_len += 1
            ob = next_ob

            if pbar is not None:
                pbar.update(1)

            if truncated:
                logger.info(f"the vehicle is truncated, now respawning")
                break  # exit collection loop

        # Post-processing: add camera_t_1 and camera_t_4
        for t in range(len(trajectory)):
            t1 = min(t + 1, len(trajectory) - 1)
            t4 = min(t + 4, len(trajectory) - 1)
            trajectory[t]['camera_t_1'] = trajectory[t1]['ob']['camera']
            trajectory[t]['camera_t_4'] = trajectory[t4]['ob']['camera']

        # Add frames to the replay buffer
        for frame in trajectory:
            self.replay_buffer.add_frame(
                obs=frame['ob'],
                rews=frame['rew'],
                terminated=frame['terminated'],
                truncated=frame['truncated'],
                info=frame['info'],
                action=frame['action'],
                closed_loop_action=frame['closed_loop_action'],
                next_state=frame['next_state'],
                camera_t_1=frame['camera_t_1'],
                camera_t_4=frame['camera_t_4'],
            )

        return traj_len

    def pretrain_save(self, evaluate_res, cur_beta):
        """Called to save ideal pretrained model that will be suitable for future experiment:
        the model that has decent generalization ability, but still not perfect"""
        completed_laps = np.asarray([evaluate_res[cur_map['name']]["completed_laps"] for cur_map in self.domain_list])
        mask = (completed_laps >= 5).astype(np.uint8)

        flag1 = np.sum(mask) == 3 and not self.pretrain_saved
        flag2 = self.cur_epoch == 6 and not self.pretrain_saved

        if flag1:
            logger.info("//// Find good pretrained object. Auto Save triggered ////")
            profile_dir = Path(__file__).parent.parent / 'training_profiles' / 'pretrain_agent_profiles'
            profile_dir.mkdir(parents=True, exist_ok=True)
            profile_path = profile_dir / f"{self.comment}_training_profile.pkl"

            self.pretrain_saved = True # save the earliest good petrained object
            self.agent.export(path=os.path.join(Path(__file__).parent / 'model_data' / 'pretrained_agents'), name=self.comment)

            profile_data = {
                'beta': cur_beta,
                'cur_epoch': self.cur_epoch,
                'evaluation_list': self.evaluation_list,  # Not serialized, just kept in structure
                'domain_list': self.domain_list,
                "model_name" : self.agent.model_name,
                'eval_domain_list': self.eval_domain_list
                }
            with open(profile_path, 'wb') as f:
                pickle.dump(profile_data, f)

    def training_loop(self, n_epochs: int):

        def make_stop_flag(consecutive_steps = 2, success_laps = 5):
            """early stop mechnism, if we have consecutive evaluations with 
            completed laps larger than success_laps, the experiment is called to early stop,
            i.e. the system is recognized to converge to stable behavior"""
            history_eval = []
            def f_stop_flag(evaluate_res):

                "find the minimum number of completed laps in all domains"
                completed_laps = min([evaluate_res[cur_map['name']]["completed_laps"] for cur_map in self.target_domains]) # only consider the target domain performances

                if completed_laps < success_laps:
                    history_eval.clear()
                else:
                    history_eval.append(True)
                
                logger.info(f"number of consecutive successfull evaluations by far: {len(history_eval)}")
                if len(history_eval) >= consecutive_steps:
                    return True
                else:
                    return False
            
            return False, f_stop_flag
            
        stop_flag, f_stop_flag = make_stop_flag()

        # Directory for saving training profiles
        profile_dir = Path(__file__).parent.parent / 'training_profiles'
        profile_dir.mkdir(parents=True, exist_ok=True)
        profile_path = profile_dir / f"{self.comment}_training_profile.pkl"

        try:
            for global_step in range(self.starting_step, n_epochs):
                logger.info(f"Epoch {global_step} / {n_epochs}")
                # self.agent.step_schedule()
                # self.sample_trajectory(beta=self.beta ** np.ceil(global_step / self.beta_decay_freq),
                #                        max_traj_len=self.initial_traj_len if global_step == 0 else self.eps_len)

                if global_step == self.mid_freeze: # check wehther to freeze the encoders or not
                    self.agent.freeze_encoders()

                cur_beta = self.init_beta * self.beta ** np.ceil(global_step / self.beta_decay_freq)
                logger.info(f"the curent beta value is {cur_beta}")
                self.sample_trajectories(beta=cur_beta,
                                         total_length=self.initial_traj_len if global_step - self.starting_step == 0 else self.eps_len,
                                         global_step=global_step)
                
                self.randomnizor.update_cur(global_step = global_step, total_epochs=n_epochs)

                self.train_module(self.agent, global_step)

                if self.no_saving:
                    continue
            
                if global_step % self.eval_freq == 0:
                    evaluate_res = self.evaluate_agent(eval_domains = self.domain_list + self.eval_domain_list)
                    # self.evaluate_randomBkg(global_step=global_step)
                    stop_flag = f_stop_flag(evaluate_res)

                    # update the evaluation list
                    for domain_name in evaluate_res.keys():
                        for benchmark in evaluate_res[domain_name].keys():
                            if benchmark not in self.evaluation_list[domain_name]:
                                self.evaluation_list[domain_name][benchmark] = []
                            self.evaluation_list[domain_name][benchmark].append(evaluate_res[domain_name][benchmark])
                    if stop_flag:
                        logger.info("//////////////////////// convergence to successful behavior  ////////////////////// early stop triggered !!!!")
                        data_dir = Path(__file__).parent.parent / 'data'
                        self.replay_buffer.export(path = data_dir, name = self.comment) # save the data only when successfully converging to successful behaviors
                        break

                    self.pretrain_save(evaluate_res, cur_beta = cur_beta)
                
                if self.latent and global_step % self.visualize_freq == 0:
                    logger.info("collecting data for latent space visualization")
                    latent_data = self.collect_latent()
                    plt.clf()
                    #plot the latent data
                    for domain_name in latent_data.keys():
                        latents = latent_data[domain_name]
                        plt.scatter(latents[:, 0], latents[:, 1], label = domain_name, s = 10)
                        plt.legend()
                    plt.show()
                
                if self.to_PCA and global_step % self.visualize_freq == 0:
                    self.PCA_visualization(self.domain_list + self.eval_domain_list)
                
                self.agent.export(path=os.path.join(Path(__file__).parent / 'model_data'), name=self.comment)
                self.cur_epoch = global_step

                # store the training profile
                if self.save_profile:
                    profile_data = {
                    'beta': cur_beta,
                    'cur_epoch': self.cur_epoch,
                    'evaluation_list': self.evaluation_list,  # Not serialized, just kept in structure
                    'domain_list': self.domain_list,
                    "model_name" : self.agent.model_name,
                    'eval_domain_list': self.eval_domain_list,
                    'pretrain_saved': self.pretrain_saved
                }
                    with open(profile_path, 'wb') as f:
                        pickle.dump(profile_data, f)
                    logger.info(f"Training profile saved to {profile_path}")

        finally:
            benchmark_list = self.evaluation_list[self.domain_list[0]["name"]].keys()

            fig, axis = plt.subplots(1, len(benchmark_list), figsize=(10, 10))

            for i, benchmark in enumerate(benchmark_list):
                axis[i].set_title(benchmark)
                for domain in self.domain_list:
                    axis[i].plot(self.evaluation_list[domain["name"]][benchmark], label = domain["name"])
                axis[i].legend()

            fig.suptitle(f"{self.agent.model_name}_{self.comment}")

            self.writer.add_figure(tag='val', figure=fig, global_step=0)

            self.writer.flush()
            logger.info(f"the images are added to logged in")
            # self.writer.ntfy(message="Training program terminated.")
    
    def PCA_visualization(self, collect_domains):
        """Using the PCA technique to visualize the high-dimensional latent vector space"""
        """Use PCA to project and visualize latent vectors from all domains in 2D."""
        logger.info("Collecting data for latent space visualization...")
        latent_data = self.collect_latent(collect_domains=collect_domains)

        # Step 1: Concatenate all latent vectors and track labels
        all_latents = []
        domain_labels = []
        domain_names = list(latent_data.keys())

        for idx, domain_name in enumerate(domain_names):
            latents = latent_data[domain_name]  # shape: [N_i, l]
            all_latents.append(latents)
            domain_labels.extend([idx] * len(latents))

        all_latents = np.vstack(all_latents)  # shape: [N_total, l]
        domain_labels = np.array(domain_labels)  # shape: [N_total]

        # Step 2: Apply PCA
        logger.info("Applying PCA projection...")
        pca = PCA(n_components=2)
        latent_2d = pca.fit_transform(all_latents)  # shape: [N_total, 2]

        # Step 3: Plot with color by domain
        plt.figure(figsize=(8, 6))
        for idx, domain_name in enumerate(domain_names):
            mask = (domain_labels == idx)
            plt.scatter(latent_2d[mask, 0], latent_2d[mask, 1], label=domain_name, s=10)

        plt.title("PCA Visualization of Latent Vectors Across Domains")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def sample_trajectories(self, beta: float, total_length=None, global_step=None):
        logger.info('Sampling trajectories for training...')
        total_length = total_length or self.eps_len
        batch_traj_len = 0
        n_resets = 0
        with tqdm(total=total_length, desc='Sampling', unit='steps') as pbar:
            while batch_traj_len < total_length:

                if global_step - self.starting_step <= 2:
                    max_traj_len = min(int(1024 / len(self.domain_list)) + 1, total_length - batch_traj_len)
                else:
                    max_traj_len = min(1024, total_length - batch_traj_len)

                sample_domain = np.random.choice(self.target_domains) # randomly choose one from the target domains
                traj_len = self.sample_trajectory(domain = sample_domain, beta = beta, pbar=pbar, max_traj_len=max_traj_len)
                batch_traj_len += traj_len
                n_resets += 1

        self.writer.do_logging({f'failure_rate': (n_resets - 1) / total_length}, global_step=global_step, mode='train')
        return batch_traj_len
    
    def sample_domain(self, global_step):
        """If the global step is the first one during training, sample domain randomly, otherwise
        sample the domain with the worst performance"""
        if global_step - self.starting_step == 0:
            return np.random.choice(self.domain_list)
        min_domain = self.domain_list[0]
        min_traj_len = self.evaluation_list[min_domain['name']]['traj_len'][-1]

        for domain in self.domain_list:
            cur_traj_len = self.evaluation_list[domain['name']]['traj_len'][-1]
            if cur_traj_len < min_traj_len:
                min_traj_len = cur_traj_len
                min_domain = domain
        
        return min_domain
    
    def evaluate_agent(self, eval_domains, global_step = 0, max_laps = 7):

        logger.info("Generalization evaluations starts")

        def one_iteration_test(domain):
            self.agent.reset()
            self.agent.eval()
            cur_map = domain["map_name"]
            weatherID = domain["weatherID"]

            ob, info = self.env.reset(options={'controller': self.expert, 'spawning': 'fixed'}, map_name =cur_map, weatherID = weatherID)
            self.expert.reset(options=info, track_obj = self.env.get_track())

            truncated, terminated = False, False
            lap_times = []
            rews = 0.
            traj_len = 0
            completed_laps = 0

            sts, acs, expert_acs, vs = [], [], [], []

            while not truncated and completed_laps <= max_laps:

                ac = self.agent.get_action(*self.agent.parse_carla_obs(ob, info))
                ob, rew, terminated, truncated, info = self.env.step(ac)
                rews += rew
                traj_len += 1
                completed_laps = info['lap_no']

                if terminated:
                    lap_times.append(info['lap_time'])

            return {"completed_laps" : completed_laps, "traj_len" : traj_len}
        
        result = {}
        logger.info("Evaluation starts")
        for domain in eval_domains:
            training_time_available = domain in self.domain_list
            title_appendix = '(training_time available)' if training_time_available else '(only evaluation time available)'
            exp_title = f"Evalution in the domain [{domain['name']}] {title_appendix}"
            logger.info(exp_title)
            result[domain["name"]] = one_iteration_test(domain)

        return result
    
    def collect_latent(self, collect_domains, max_laps = 1):

        logger.info("Collecting latent vectors")

        def one_domain_collect(domain):
            self.agent.reset()
            self.agent.eval()
            cur_map = domain["map_name"]
            weatherID = domain["weatherID"]

            ob, info = self.env.reset(options={'controller': self.expert, 'spawning': 'fixed'}, map_name =cur_map, weatherID = weatherID)
            self.expert.reset(options=info, track_obj = self.env.get_track())

            truncated, terminated = False, False
            lap_times = []
            rews = 0.
            traj_len = 0
            completed_laps = 0

            latent_list = []

            while not truncated and completed_laps <= max_laps:

                expert_ac, expert_info = self.expert.step(**ob, **info)
                expert_ac = np.clip(expert_ac, self.env.action_space.low, self.env.action_space.high)
                ob, rew, terminated, truncated, info = self.env.step(expert_ac)
                l = self.agent.get_latent(ptu.from_numpy(ob["camera"].copy()[None]))
                latent_list.append(l)
                rews += rew
                traj_len += 1
                completed_laps = info['lap_no']

                if terminated:
                    lap_times.append(info['lap_time'])

            return np.concatenate(latent_list, axis = 0)
        
        result = {}
        for domain in collect_domains:
            logger.info(f"collecting latent vectors in the domain [{domain['name']}]")
            result[domain["name"]] = one_domain_collect(domain)

        return result

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=37) # original seed 38
    parser.add_argument('--n_epochs', type=int, default=500)
    parser.add_argument('--initial_traj_len', type=int, default=3072)
    parser.add_argument('--n_training_per_epoch', type=int, default=1)
    parser.add_argument('--n_initial_training_epochs', type=int, default=5)
    parser.add_argument('--replay_buffer_maxsize', type=int, default= 51_200) # the original replay buffer maximum size: 102_400
    parser.add_argument('--expert', '-c', type=str, default='mpcc-conv',
                        choices=tuple(expert_mp.keys()))
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--comment', '-m', type=str, default='')
    parser.add_argument('--eps_len', type=int, default=1024)
    parser.add_argument('--randomnize', default = '', choices = ('', 'pure_augment', "random_fetch", "contrast"))

    parser.add_argument('--town', type=str, default='L_track_barc')
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--no_saving', action='store_true')
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--beta', type=float, default=0.90)
    parser.add_argument('--beta_decay_freq', type=int, default=1)

    parser.add_argument('--evaluation', action='store_true')
    parser.add_argument('--continue', type=int, default=0)
    parser.add_argument('--freeze_weather', action='store_true')
    parser.add_argument('--fix_spawning', action='store_true')
    parser.add_argument('--pretrain_critic', action='store_true')
    parser.add_argument('--observe', '-o', type=str, default='camera',
                        choices=('camera', 'state'))

    parser.add_argument('--experimental', action='store_true')
    parser.add_argument("--generalize_test", action = 'store_true')
    parser.add_argument("--data_collect", action = 'store_true')
    parser.add_argument("--save_profile", action = "store_true", default = True) # whether to save thet training profiles
    parser.add_argument("--reload", action = "store_true", default = False)# whether to reload the existing model with the same name to keep training
    parser.add_argument("--latent", action = "store_true", default = False) # wether to visualize the latent vectors or not
    parser.add_argument("--mid_freeze", type = int, default = np.inf) # at what point to freeze the encoders
    parser.add_argument("--to_PCA", action = 'store_true', default = False) # whether to use the PCA technique to visualize the latent vector space
    parser.add_argument("--save_data", action = 'store_true', default = True) # whether to save the data buffer or not

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
    if params['evaluation']:
        params['reload'] = True # if the user wants to do evaluation, always reload the parameters

    params['comment'] = '_'.join((params['town'], params['comment']))

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

    config_path = "./config/VisionAttentionActor.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    agent_params = config['model_hparams']
    trainer_cls = IL_Trainer_CARLA_Perfect_Baseline

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
                          save_profile = params["save_profile"],
                          to_reload = params["reload"],
                          latent = params["latent"],
                          mid_freeze = params["mid_freeze"],
                          to_PCA = params["to_PCA"],
                          save_data = params['save_data'],
                            **agent_params
                          )

    if params['evaluation']:
        logger.info(f"the evaluated agent has been trained for {trainer.starting_step} epochs")
        trainer.evaluate_agent(eval_domains=FULL_EVALUATION_LIST, global_step=0)
        trainer.PCA_visualization(collect_domains = FULL_EVALUATION_LIST)

    elif params["generalize_test"]:
        trainer.agent.load(path=Path(__file__).resolve().parent / 'model_data',
                           name=comment)
        trainer.gnz_evaluation()
    
    elif params["data_collect"]:
        trainer.data_collect()

    else:
        trainer.main(n_epochs=params['n_epochs'])