"""The trainer class for the conditional adversarial domain adaptation,
a pretrained model will be loaded together with its data_loader.
RGB images will be sampled from the target domain without aay expert supervision available"""

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
from models.TCADT import VisionNaiveRandomization, VisionAdversarialAdaptAC, VisionAdversarialActor, VisionConditionAdversarialAdaptAC
from models.base_model import BaseModel
from domain_randomnization.randomnizor import BkgRandomnizer

from utils import data_util
from torch.utils.data import DataLoader
import utils.pytorch_util as ptu
from utils.logging.writer import MultiPurposeWriter
import utils.plot_util as ptl
import pickle
from sklearn.decomposition import PCA

from loguru import logger
from labml import experiment

"""Listed below are available map names"""
L_TRACK_BARC = "L_track_barc" # the original map without any additional features

#available pretrained agents
DEMO1 = "demo1"

expert_mp = {
    'pid': PIDWrapper,
    'mpcc-conv': MPCCConvWrapper,
}
domain_config_path = "./config/domain_config.yaml"
with open(domain_config_path, 'r') as f:
    domain_config = yaml.safe_load(f)
domains = domain_config["domains"] # read the hard-coded domains from the domain_config file

class IL_Trainer_CARLA_VisionAdversarialAdaptationAC():

    """classifier is now considered as part of the architecture"""

    def __init__(self, carla_params,
                       expert_cls,
                       augment_percent=0.6,
                       initial_traj_len=4096,
                       eps_len=1024,
                       replay_buffer_maxsize=65536,
                       do_relabel_with_expert=True,
                       n_training_per_epoch=1,
                       comment='',
                       starting_step=0,
                       eval_freq=1,
                       batch_size=1,
                       n_initial_training_epochs=5,
                       beta=0.25,
                       beta_decay_freq=5,
                       save_model = True,
                       target_domain_len = 5000,
                       source_buffer = None,
                       env = None,
                       visualize = True,
                       sample_distribution = 'naive_random',
                       target_domains = [domains["DOMAIN11"]],
                       source_domains = [],
                       pretrain_agent = "demo1",
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
        @param starting_step:
        @param eval_freq:
        @param batch_size:
        @param n_initial_training_epochs:
        @param agent_params:
        """
        self.carla_params = carla_params
        
        self.pretrain_encoder_path = pretrain_agent
        self.target_domain_len = target_domain_len
        self.target_domains = target_domains
        
        self.domain_list = source_domains
        
        self.save_model = save_model
        self.visualize = visualize
        self.sample_distribution = sample_distribution

        self.best_avg_lap_time = np.inf
        self.beta_decay_freq = beta_decay_freq
        self.eval_freq = eval_freq
        self.comment = comment
        self.initial_traj_len = initial_traj_len
        self.do_relabel_with_expert = do_relabel_with_expert
        self.n_training_per_epoch = n_training_per_epoch
        self.batch_size = batch_size
        self.n_initial_training_epochs = n_initial_training_epochs
        self.beta = beta
        # self.use_labml_tracker = use_labml_tracker
        self.agent_params = agent_params

        self.n_eval_success, self.n_eval_total = 0, 0
        self.eval_rewards_last10 = deque(maxlen=10)
        self.visualize_freq = 2 # the frequency of visualizing latent vectors in terms epoc num

        # the parameters for randomizer
        self.randomnizor = BkgRandomnizer(transfer_percentage = augment_percent)
        transform = {"camera": self.randomnizor.traditional_randomnize}

        #initialize the data buffer
        self.replay_buffer: 'data_util.sourceTargetBalanceBuffer' = None
        self.replay_buffer = data_util.sourceTargetBalanceBuffer(maxsize=replay_buffer_maxsize,
                                        lazy_init=True,
                                        transform = transform,
                                        source_buffer = source_buffer)
        
        data_dir = Path(__file__).parent.parent / 'data'
        # only load the source buffer if no source buffer is given as a parameter
        if source_buffer is None:
            self.replay_buffer.source_buffer.load(path = data_dir, name = self.pretrain_encoder_path)

        if env is None:
            self.env = gym.make('barc-v0', **carla_params)
            self.eps_len = min(replay_buffer_maxsize, eps_len)
        else:
            self.env = env

        self.expert = expert_cls(dt=carla_params['dt'], t0=carla_params['t0'], track_obj=self.env.get_track())
        self.agent: 'BaseModel' = None
        self.init_beta = 1.0

        # Load previous model weights and replay buffer.
        self.starting_step = starting_step

        self.initialize_agent(comment=comment, ad_agent_params = agent_params)

        self.writer: 'MultiPurposeWriter' = MultiPurposeWriter(model_name=self.agent.model_name,
                                                               log_dir=f"logs/{self.agent.model_name}_{comment or ''}",
                                                               comment=comment or '',
                                                               print_method=logger.info,
                                                               )
            
    def initialize_agent(self, comment, ad_agent_params):
        pretrain_config_path = "./config/VisionNaiveRandomization.yaml"
        with open(pretrain_config_path, 'r') as f:
            config = yaml.safe_load(f)

        pretrain_agent_params = config['model_hparams']

        # modify the learning rate of the pretrained agent for better fine tuning
        pretrain_agent_params['lr'] = ad_agent_params['lr_actor']
        pretrain_agent = VisionNaiveRandomization(**pretrain_agent_params)

        logger.info("//// Loading pretrained encoders ////")
        
        if self.pretrain_encoder_path == "null":
            logger.info("No pretrained agent is passed in. Training the agent from scratch")
            
        if self.pretrain_encoder_path != "null": # if the pretrain encoder path is null skip loading, just train it from scratch
            pretrain_agent.load(path=Path(__file__).resolve().parent / 'model_data',
                            name=self.pretrain_encoder_path)
        
        logger.info(f"The pretrained agent is warmed up in the domains {pretrain_agent.domain_list}")
        
        #read the pretrained agent's source domains and set it's source domain list as trainer attribute 
        logger.debug(pretrain_agent.domain_list)
        # self.domain_list += pretrain_agent.domain_list # add the pretrain

        actor_cls = VisionConditionAdversarialAdaptAC
        
        self.agent = actor_cls(pretrain_agent = pretrain_agent, pretrain_agent_params = pretrain_agent_params, ad_agent_params = ad_agent_params)
        logger.debug(f"the loaded model is {self.agent.model_name}")

    def initialize_replay_buffer():
        return None
    
    def train_module(self, module, global_step):
        logger.info(f"Training {module.__class__.__name__}...")
        info = module.fit(train_dataset=self.replay_buffer,
                          n_epochs= self.n_training_per_epoch if global_step > 0 else self.n_initial_training_epochs,
                          global_step=global_step)
        
        # for mode, values in info.items():
        #     self.writer.do_logging(values, global_step=global_step, mode=mode)
    
    def label_domain(self, ob, domain):
        """add the field 'domain_v', the domain probability vector to the ob
         @ob: the observation dictionary returned from the carla-gym environment
         @domain: the domain we used to so far"""
        domain_indicator = np.asarray([0.]) # set the indicator to zero if the domaindoes not belong the training_time available domains
        if domain in self.domain_list: #TODO: when loading pretrain-agents, how to keep track of their source domain list
            domain_indicator = np.asarray([1.])
        ob['domain_indicator'] = domain_indicator
        return ob

    def sample_trajectory(self, domain, beta: float, pbar: Optional['tqdm'] = None,
                          max_traj_len=np.inf,
                          PATIENCE=2, TRUNCATE=np.inf, buffer: data_util.EfficientReplayBuffer = None):
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
        ob, info = self.env.reset(options={'controller': self.expert}, map_name = cur_map, weatherID = weatherID)
        self.agent.reset()
        self.agent.eval()
        self.expert.reset(options=info)
        terminated, truncated = False, False
        traj_len = 0
        fail_counter = 0

        while traj_len < max_traj_len:

            self.label_domain(ob, domain) # label the carla observation the domain probability vector
            ac, _ = self.agent.get_action(*self.agent.parse_carla_obs(ob, info))
            expert_ac, expert_info = self.expert.step(**ob, **info)
            expert_ac = np.clip(expert_ac, self.env.action_space.low, self.env.action_space.high)
            closed_loop_action = beta * expert_ac + (1 - beta) * ac

            # try:
            if expert_info['success']:
                next_ob, rew, terminated, truncated, info = self.env.step(closed_loop_action)
                buffer.add_frame(ob, rew, terminated, truncated, info,
                                             action=expert_ac.astype(np.float32),
                                             closed_loop_action=closed_loop_action.astype(np.float32),
                                             next_state=next_ob['state'])
                fail_counter = 0

            else:
                logger.warning(f"Expert solved inaccurate with code {expert_info.get('status', 'unknown')}.")
                next_ob, rew, terminated, truncated, info = self.env.step(closed_loop_action)
                fail_counter += 1
                buffer.add_frame(ob, rew, terminated, truncated, info,
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
                ob, info = self.env.reset(options={'controller': self.expert})
                self.agent.reset()
                self.agent.eval()
                self.expert.reset(options=info)
                terminated, truncated = False, False

        return traj_len
    
    def sample_pid_trajectory(self, domain, pbar: Optional['tqdm'] = None,
                          max_traj_len=np.inf,
                          PATIENCE=2, TRUNCATE=np.inf, buffer: data_util.EfficientReplayBuffer = None):
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
        pid_expert = PIDWrapper(dt=self.carla_params['dt'], t0=self.carla_params['t0'], track_obj=self.env.get_track())
        ob, info = self.env.reset(options={'controller': pid_expert}, map_name = cur_map, weatherID = weatherID)
        pid_expert.reset(options=info)

        terminated, truncated = False, False
        traj_len = 0
        fail_counter = 0

        while traj_len < max_traj_len:

            self.label_domain(ob, domain) # label the carla observation the domain probability vector
            expert_ac, expert_info = pid_expert.step(**ob, **info)
            expert_ac = np.clip(expert_ac, self.env.action_space.low, self.env.action_space.high)

            # try:
            if expert_info['success']:
                next_ob, rew, terminated, truncated, info = self.env.step(expert_ac)
                buffer.add_frame(ob, rew, terminated, truncated, info,
                                             action=expert_ac.astype(np.float32),
                                             closed_loop_action=expert_ac.astype(np.float32),
                                             next_state=next_ob['state'])
                fail_counter = 0

            else:
                logger.warning(f"Expert solved inaccurate with code {expert_info.get('status', 'unknown')}.")
                next_ob, rew, terminated, truncated, info = self.env.step(expert_ac)
                fail_counter += 1
                buffer.add_frame(ob, rew, terminated, truncated, info,
                                                   action=expert_ac.astype(np.float32),
                                                   closed_loop_action=expert_ac.astype(np.float32),
                                                   next_state=next_ob['state'])
                if fail_counter >= PATIENCE:
                    truncated = True

            traj_len += 1
            ob = next_ob

            if pbar is not None:
                pbar.update(1)
            
            if truncated: # reset the environment if the vehicle is truncated
                ob, info = self.env.reset(options={'controller': pid_expert})
                pid_expert.reset(options=info)
                terminated, truncated = False, False

        return traj_len

    def random_sample(self, domain,  pbar: Optional['tqdm'] = None, max_traj_len = np.inf, buffer: data_util.EfficientReplayBuffer = None):
        cur_map = domain["map_name"]
        weatherID = domain["weatherID"]

        ob, info = self.env.reset(options={}, map_name=cur_map, weatherID=weatherID)

        self.agent.reset()
        self.agent.eval()
        self.expert.reset(options=info)

        terminated, truncated = False, False
        traj_len = 0

        while traj_len < max_traj_len:
            self.label_domain(ob, domain)
            if self.sample_distribution == 'naive_random':
                next_ob = self.env.naive_random_obs()
            elif self.sample_distribution == 'first_4m_random':
                next_ob = self.env.first4mDistribution_random_obs() # choose the random sample distribution policy
            elif self.sample_distribution == 'middle_3m_random':
                next_ob = self.env.middle3mDistribution_random_obs() 
            else:
                raise ValueError(f"the sample_distribution {self.sample_distribution} is not available")

            #some null observation without specific meaning but only for data structure consistency
            rew, expert_ac, closed_loop_action = 0.0, np.zeros((2,)), np.zeros((2,))

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

            buffer.add_frame(
                obs=frame['ob'],
                rews=frame['rew'],
                terminated=frame['terminated'],
                truncated=frame['truncated'],
                info=frame['info'],
                action=frame['action'],
                closed_loop_action=frame['closed_loop_action'],
                next_state=frame['next_state'],
            )

            traj_len += 1
            ob = next_ob

            if pbar is not None:
                pbar.update(1)

        return traj_len
    
    def early_stop(self, evaluate_res) -> bool:
        """judge whether to early stop"""
        "find the minimum number of completed laps in all domains"
        completed_laps = min([evaluate_res[domain_name]["completed_laps"] for domain_name in evaluate_res.keys()])
        return completed_laps >= 5
    
    def training_loop(self, n_epochs: int):
        max_traj_len = 0

        visualization_list = [self.domain_list[0], self.target_domains[0]]
        
        # before starting tuning the model for domain adpatation, do the evaluation for different domains
        logger.info("Pretraining Evaluation .......")
        self.evaluate_agent(eval_domains = self.target_domains)
        if self.visualize:
            self.PCA_visualization(visualization_list, display_full_name = False)

        # Directory for saving training profiles
        profile_dir = Path(__file__).parent.parent / 'training_profiles'
        profile_dir.mkdir(parents=True, exist_ok=True)
        profile_path = profile_dir / f"{self.comment}_training_profile.pkl"
        
        logger.info("---------- collecting data from the target domains ----------")
        
        try:
            cur_beta = 0
            self.target_sample(beta = cur_beta, domain_list = self.target_domains,
                    total_length=self.target_domain_len, buffer = self.replay_buffer.target_buffer,
                    global_step=0, sample_policy=self.random_sample)
            
        finally:
            logger.info("------- data collecting from the target domain ends ---------")
            
        """start the agent domain adaptation"""
        logger.info("--------------Adversarial Domain Adaptation starts-------------")

        stop_flag = False
        
        for global_step in range(self.starting_step, n_epochs):
            logger.info(f"Epoch {global_step} / {n_epochs}")
            cur_beta = self.init_beta * (self.beta ** (global_step - self.starting_step))
            self.sample_trajectories(beta=cur_beta,
                                     domain_list = self.domain_list,
                                         total_length=self.eps_len,
                                         buffer = self.replay_buffer.source_buffer,
                                         global_step=global_step)
            
            self.train_module(self.agent, global_step)
            logger.info(f"the current beta is {cur_beta}")

            if global_step % self.visualize_freq == 0 and self.visualize:
                self.PCA_visualization(visualization_list, display_full_name=False)
                
            if global_step % self.eval_freq == 0:
                evaluate_res = self.evaluate_agent(eval_domains = self.target_domains, global_step = global_step) # only evaluate the agent's performance in the target domain
                max_traj_len = max(max_traj_len, evaluate_res[self.target_domains[0]['name']]['traj_len'])

                # self.evaluate_randomBkg(global_step=global_step)
                stop_flag = self.early_stop(evaluate_res)

                if stop_flag:
                    logger.info("Final visualization for the successfull convergence ...")
                    if self.visualize:
                        self.PCA_visualization(visualization_list)
                    logger.info("//////////////////////// convergence to successful behavior  ////////////////////// early stop triggered !!!!")
                    break

        if self.save_model:
            self.agent.export(path=os.path.join(Path(__file__).parent / 'model_data'), name=self.comment)

        logger.info(f"the maximum achived trajectory length in the target domain system = {max_traj_len}")
        return max_traj_len

    def PCA_visualization(self, collect_domains, display_full_name=True):
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
            if display_full_name:
                legend_name = domain_name
            else:
                source_name_list = [domain['name'] for domain in self.domain_list]
                if domain_name in source_name_list:
                    domain_index = source_name_list.index(domain_name)
                    legend_name = f"source domain {domain_index}"
                else:
                    legend_name = "target domain"
            plt.scatter(latent_2d[mask, 0], latent_2d[mask, 1], label=legend_name, s=10)

        plt.title("PCA Visualization of Latent Vectors Across Domains")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def collect_latent(self, collect_domains, max_laps = 1):

        logger.info("Collecting latent vectors")

        def one_domain_collect(domain):
            self.agent.reset()
            self.agent.eval()
            cur_map = domain["map_name"]
            weatherID = domain["weatherID"]

            ob, info = self.env.reset(options={'controller': self.expert, 'spawning': 'fixed'}, map_name =cur_map, weatherID = weatherID)
            self.expert.reset(options=info)

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

    def target_sample(self, domain_list, beta: float, total_length=None, global_step=None, buffer: data_util.EfficientReplayBuffer = None, sample_policy = None):
        logger.info('Sampling trajectories for training...')
        total_length = total_length or self.eps_len
        batch_traj_len = 0
        n_resets = 0
        with tqdm(total=total_length, desc='Sampling', unit='steps') as pbar:
            while batch_traj_len < total_length:

                max_traj_len = min(1024, total_length - batch_traj_len)
                rand_beta = np.random.uniform(low = beta - 0.1, high = beta + 0.1)
                domain = np.random.choice(domain_list)

                #traj_len = self.sample_trajectory(domain = domain, beta = rand_beta, pbar=pbar, max_traj_len=max_traj_len, buffer = buffer)
                traj_len = sample_policy(domain = domain, pbar = pbar, max_traj_len=max_traj_len, buffer = buffer)

                batch_traj_len += traj_len
                n_resets += 1

        self.writer.do_logging({f'failure_rate': (n_resets - 1) / total_length}, global_step=global_step, mode='train')
        return batch_traj_len
    
    def sample_trajectories(self, domain_list, beta: float, total_length=None, global_step=None, buffer: data_util.EfficientReplayBuffer = None):
        logger.info('Sampling trajectories for training...')
        total_length = total_length or self.eps_len
        batch_traj_len = 0
        n_resets = 0
        with tqdm(total=total_length, desc='Sampling', unit='steps') as pbar:
            while batch_traj_len < total_length:

                max_traj_len = min(1024, total_length - batch_traj_len)
                domain = np.random.choice(domain_list)

                traj_len = self.sample_trajectory(domain = domain, beta = beta, pbar=pbar, max_traj_len=max_traj_len, buffer = buffer)
                
                batch_traj_len += traj_len
                n_resets += 1
                
        self.writer.do_logging({f'failure_rate': (n_resets - 1) / total_length}, global_step=global_step, mode='train')
        return batch_traj_len

    def evaluate_agent(self, eval_domains, global_step = 0, max_laps = 4):

        logger.info("Generalization evaluations starts")

        def one_iteration_test(eval_domain):
            self.agent.reset()
            self.agent.eval()
            cur_map = eval_domain["map_name"]
            weatherID = eval_domain["weatherID"]

            ob, info = self.env.reset(options={'controller': self.expert, 'spawning': 'fixed'}, map_name =cur_map, weatherID = weatherID)

            truncated, terminated = False, False
            lap_times = []
            rews = 0.
            traj_len = 0
            completed_laps = 0
            domain_probs_list = []

            while not truncated and completed_laps <= max_laps:

                self.label_domain(ob, eval_domain)
                ac, domain_probs_pred = self.agent.get_action(*self.agent.parse_carla_obs(ob, info))
                domain_probs_list.append(domain_probs_pred)
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
            exp_title = f"Evalution in the domain {domain['name']}"
            logger.info(exp_title)
            result[domain["name"]] = one_iteration_test(domain)
        
        return result


    def traj_collect(self, eval_domain, actor):

        cur_map = eval_domain["map_name"]
        weatherID = eval_domain["weatherID"]

        ob, info = self.env.reset(options={'controller': self.expert, 'spawning': 'fixed'}, map_name =cur_map, weatherID = weatherID)

        if actor is self.agent:
            actor.reset()
            actor.eval()
        else:
            actor.reset(options = info)

        truncated, terminated = False, False
        rews = 0.
        traj_len = 0
        completed_laps = 0

        traj = []

        while not truncated and completed_laps <= 1:

            self.label_domain(ob, eval_domain)
            if actor is self.agent:
                ac, domain_probs_pred = actor.get_action(*self.agent.parse_carla_obs(ob, info))
            else:
                ac, actor_info = actor.step(**ob, **info)

            ob, rew, terminated, truncated, info = self.env.step(ac)
            rews += rew
            traj_len += 1
            completed_laps = info['lap_no']

            traj.append(np.concatenate([ob['gps'][ : 2], ob['velocity'][ : 2]]))
        return traj

    def speed_regime_plot(self):
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable, get_cmap

        # collect the trajectories

        eval_domain = domains["DOMAIN11"]
        traj_agent_no_trans = self.traj_collect(eval_domain = eval_domain, actor = self.agent) # the agent trajectory before domain transfer

        self.agent.load(path=Path(__file__).resolve().parent / 'model_data',
                           name=comment)
        
        traj_agent_after_trans = self.traj_collect(eval_domain = eval_domain, actor = self.agent) # the agent trajectory after domain transfer
        traj_collector = self.traj_collect(eval_domain = eval_domain, actor = PIDWrapper(dt=self.carla_params['dt'], t0=self.carla_params['t0'], track_obj=self.env.get_track())) # the data collector's trajectory
        traj_expert = self.traj_collect(eval_domain = eval_domain, actor = self.expert) # the expert trajectory

        trajs = [
        ("Agent (before transfer)", traj_agent_no_trans),
        ("Agent (after transfer)",  traj_agent_after_trans),
        ("Safe Collector (PID)",    traj_collector),
        ("Expert",                  traj_expert),
    ]
        # ------------------------------
        # 2. Compute global speed range
        # ------------------------------
        all_speeds = []
        for _, arr in trajs:
            arr = np.asarray(arr, dtype=object)
            for A in arr:
                A = np.asarray(A)
                if A.ndim > 1 and A.shape[1] >= 4:
                    v = np.linalg.norm(A[:, 2:4], axis=1)
                    all_speeds.append(v)
                elif A.ndim == 1 and A.size >= 4:
                    v = np.linalg.norm(A[2:4])
                    all_speeds.append([v])

        all_speeds = np.concatenate(all_speeds)
        vmin, vmax = np.nanmin(all_speeds), np.nanmax(all_speeds)
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = get_cmap('plasma')

        # ------------------------------
        # 3. Plot trajectories
        # ------------------------------
        fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
        axes = axes.ravel()

        # robust access to track object
        get_trak = getattr(self.env, "get_trak", None)
        track_obj = get_trak() if callable(get_trak) else self.env.get_track()

        for ax, (title, traj) in zip(axes, trajs):
            track_obj.plot_map(ax)
            ax.set_aspect('equal')
            ax.set_title(title)
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.5)

            # draw with your helper
            sc = ptl.plot_speed_scatter(ax, traj, cmap=cmap, add_colorbar=False)

            # 🔧 ensure consistent mapping across ALL subplots
            sc.set_norm(norm)
            sc.set_cmap(cmap)

        # ------------------------------
        # 4. Shared colorbar
        # ------------------------------
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, fraction=0.035, pad=0.02)
        cbar.set_label("Speed magnitude [m/s]")

        plt.show()
        return fig, axes

    def visualize_distributions(self, distributions=('naive_random',
                                                 'first_4m_random',
                                                 'middle_3m_random')):
        """
        Collect independent data buffers based on the passed-in distributions.
        For each collected data buffer, create a subplot visualizing the density
        of visited states inside the track.
        """
        # 1) Collect data buffers
        demo_buffers = {}
        for distribution in distributions:
            demo_buffers[distribution] = data_util.EfficientReplayBuffer(
                maxsize=4000,
                lazy_init=True
            )

        for distribution in distributions:
            logger.debug(f"Collecting data for {distribution}")
            self.sample_distribution = distribution
            # you already had this line:
            self.random_sample(
                domain=domains["DOMAIN11"],
                max_traj_len=2048,
                buffer=demo_buffers[distribution]
            )

        # 2) Get track object using your API
        get_trak = getattr(self.env, "get_trak", None)
        track_obj = get_trak() if callable(get_trak) else self.env.get_track()

        # 3) Set up subplots
        n = len(distributions)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
        axes = axes[0]  # flatten 1 x n

        # 4) For each distribution, extract (x, y) and plot density over the track
        for ax, dist_name in zip(axes, distributions):
            buffer = demo_buffers[dist_name]

            # ---- extract states from buffer ----
            # Assumes: buffer[i] -> dict with key "state", and state[0], state[1] = x, y
            xs = []
            ys = []
            for i in range(len(buffer)):
                transition = buffer[i]          # may need to adapt to your API
                state = transition["gps"]     # e.g., shape (state_dim,)
                xs.append(state[0])             # x position
                ys.append(state[1])             # y position

            xs = np.array(xs)
            ys = np.array(ys)

            # ---- plot track as background ----
            track_obj.plot_map(ax)

            # ---- overlay density of samples ----
            # You can tweak bins/cmap as you like
            if len(xs) > 0:
                h = ax.hist2d(xs, ys, bins=50, density=True,
                            alpha=0.8)  # semi-transparent heatmap
                # Optional: add a colorbar per subplot
                fig.colorbar(h[3], ax=ax, fraction=0.046, pad=0.04)

            ax.set_aspect('equal')
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.5)

        plt.tight_layout()
        plt.show()
            
    def main(self, n_epochs: int):
        self.visualize_distributions()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=37) # original seed 38
    parser.add_argument('--n_epochs', type=int, default=500)
    parser.add_argument('--initial_traj_len', type=int, default=3072)
    parser.add_argument('--n_training_per_epoch', type=int, default=1)
    parser.add_argument('--n_initial_training_epochs', type=int, default=1) 
    parser.add_argument('--replay_buffer_maxsize', type=int, default= 51_200) # the original replay buffer maximum size: 102_400
    parser.add_argument('--expert', '-c', type=str, default='pid',
                        choices=tuple(expert_mp.keys()))
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--comment', '-m', type=str, default='')
    parser.add_argument('--eps_len', type=int, default=1024)
    parser.add_argument('--randomnize', default = '', choices = ('', 'pure_augment', "random_fetch", "contrast"))

    parser.add_argument('--town', type=str, default='L_track_barc')
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--beta', type=float, default=0.8)
    parser.add_argument('--beta_decay_freq', type=int, default=1)

    parser.add_argument('--evaluation', action='store_true')
    parser.add_argument('--continue', type=int, default=0)
    parser.add_argument('--freeze_weather', action='store_true')
    parser.add_argument('--fix_spawning', action='store_true')
    parser.add_argument('--observe', '-o', type=str, default='camera',
                        choices=('camera', 'state'))

    parser.add_argument('--experimental', action='store_true')
    parser.add_argument("--generalize_test", action = 'store_true')
    parser.add_argument("--data_collect", action = 'store_true')
    parser.add_argument("--reload", action = "store_true", default = False)# whether to reload the existing model with the same name to keep training
    # parser.add_argument('--ntfy_freq', type=int, default=100)
    parser.add_argument("--target_domain_len", '-t', type = int, default = 2048, help = "the size of the target-domain buffer") # the length of total trajectory sampled from the target domain
    parser.add_argument("--dis_info_mode", '-d', type = str, default = "state_curvature", choices = ('only_curvature', 'state_curvature', 'only_state', 'only_x_tran', 'gps_xy', 'gps_full'), help='the discriminative information used for domain transfer')
    parser.add_argument("--target_sample_distribution", '-ts', type = str, default = 'naive_random', choices = ('naive_random', 'first_4m_random', 'middle_3m_random'), help = 'the distribution used to collect data in the target domain')
    parser.add_argument(
        "-td", "--target_domains",
        nargs="*",
        type=str,
        default=[],
        help="List of target domain names",
    )
    parser.add_argument(
        "-sd", '--source_domains',
        nargs="*",
        type=str,
        default=[],
        help="List of source domain names",
    )
    parser.add_argument("--pretrain_agent", "-p", type = str, default = "null", help = 'The name of the pretrain agent. If you want to train an agent from scratch, pass in "null".')
    parser.add_argument("--speed_plot", action = 'store_true', default = False)

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

    params['comment'] = params["comment"]

    t0, dt, dt_sim = 0., 0.1, 0.01

    if params['evaluation']:
        params['reload'] = True # set the trainer to reload old parameters if the user wants to do evaluation

    carla_params = dict(
        track_name=params['town'],
        t0=t0, dt=dt, dt_sim=dt_sim,
        do_render=params['render'],
        max_n_laps=50,
        enable_camera=params['observe'] == 'camera',
        host=params['host'],
        port=params['port'],
    )

    config_path = "./config/VisionCAD.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    agent_params = config['model_hparams']
    agent_params["dis_info_mode"] = params["dis_info_mode"]
    
    trainer_cls = IL_Trainer_CARLA_VisionAdversarialAdaptationAC
    
    try:
        target_domains = []
        source_domains = []
        for target_domain in params["target_domains"]:
            target_domains.append(domains[target_domain])
        for source_domain in params["source_domains"]:
            source_domains.append(domains[source_domain])
    
    except KeyError:
        raise KeyError(f"Make sure the source domains and the target domains belong to the following list of available domains: {domains.keys()}")
    
    trainer = trainer_cls(carla_params,
                          expert_cls=expert_mp[params['expert']],
                          replay_buffer_maxsize=params['replay_buffer_maxsize'],
                          eps_len=params['eps_len'],
                          initial_traj_len=params['initial_traj_len'],
                          do_relabel_with_expert=True,
                          n_training_per_epoch=params['n_training_per_epoch'],
                          comment=params['comment'],
                          starting_step=params['continue'],
                          eval_freq=params['eval_freq'],
                          batch_size=params['batch_size'],
                          n_initial_training_epochs=params['n_initial_training_epochs'],
                          beta=params['beta'],
                          beta_decay_freq=params['beta_decay_freq'],
                          target_domain_len = params["target_domain_len"],
                          sample_distribution = params['target_sample_distribution'],
                          target_domains = target_domains,
                          pretrain_agent = params["pretrain_agent"],
                          source_domains = source_domains,
                          **agent_params
                          )

    if params['evaluation']:
        logger.info(f"the evaluated agent has been trained for {trainer.starting_step + trainer.pretrain_agent_epochs} epochs")
        trainer.evaluate_agent(eval_domains = [domains[domain_label] for domain_label in domains.keys()], global_step=0)

    elif params["generalize_test"]:
        trainer.agent.load(path=Path(__file__).resolve().parent / 'model_data',
                           name=comment)
        trainer.gnz_evaluation()
    
    elif params["speed_plot"]:
        trainer.speed_regime_plot()
    
    elif params["data_collect"]:
        trainer.data_collect()

    else:
        trainer.main(n_epochs=params['n_epochs'])