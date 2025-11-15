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
from models.base_model import BaseModel
from domain_randomnization.randomnizor import BkgRandomnizer

from utils import data_util
from torch.utils.data import DataLoader
import utils.pytorch_util as ptu
from utils.logging.writer import MultiPurposeWriter
import pickle
from sklearn.decomposition import PCA

from loguru import logger
from labml import experiment
import models.TCADT

L_TRACK_BARC = "L_track_barc" # the original map without any additional features

domain_config_path = "./config/domain_config.yaml"
with open(domain_config_path, 'r') as f:
    domain_config = yaml.safe_load(f)
domains = domain_config["domains"] # read the hard-coded domains from the domain_config file

expert_mp = {
    'pid': PIDWrapper,
    'mpcc-conv': MPCCConvWrapper,
}

class IL_Trainer_CARLA_VisionNaiveRandomizationAC():

    def __init__(self, carla_params,
                       expert_cls,
                       augment_percent=0.6,
                       initial_traj_len=1024,
                       eps_len=1024,
                       replay_buffer_maxsize=65536,
                       do_relabel_with_expert=True,
                       n_training_per_epoch=1,
                       comment='',
                       eval_freq=1,
                       batch_size=1,
                       n_initial_training_epochs=5,
                       beta=0.25,
                       beta_decay_freq=5,
                       save_data = True,
                       domain_list = [domains["DOMAIN4"], domains["DOMAIN5"]],
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
        @param eval_freq:
        @param batch_size:
        @param n_initial_training_epochs:
        @param agent_params:
        """
        self.domain_list = domain_list # the training-time available domains
        self.cur_epoch = 0
        self.beta_decay_freq = beta_decay_freq
        self.eval_freq = eval_freq
        self.comment = comment
        self.initial_traj_len = initial_traj_len
        self.do_relabel_with_expert = do_relabel_with_expert
        self.n_training_per_epoch = n_training_per_epoch
        self.batch_size = batch_size
        self.n_initial_training_epochs = n_initial_training_epochs
        self.init_beta = 1.0
        self.beta = beta
        # self.use_labml_tracker = use_labml_tracker
        self.agent_params = agent_params

        self.n_eval_success, self.n_eval_total = 0, 0
        self.eval_rewards_last10 = deque(maxlen=10)

        self.env = gym.make('barc-v0', **carla_params)
        self.eps_len = min(replay_buffer_maxsize, eps_len)

        self.expert_cls = expert_cls
        self.carla_params = carla_params

        self.expert = self.expert_cls(dt=self.carla_params['dt'], t0=self.carla_params['t0'], track_obj=self.env.get_track())
        self.agent: 'BaseModel' = None
        self.save_data = save_data

        self.initialize_agent(**agent_params)

        #set the randomization
        self.randomnizor = BkgRandomnizer(transfer_percentage = augment_percent) # set debug to false to speed up rendering
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
        self.agent.to(ptu.device)
        self.eval_domain_list = [] # the additional domains other than trianing-time available domains used for evaluation only
        
    
    def label_domain(self, ob, domain):
        """add the field 'domain_v', the domain probability vector to the ob
         @ob: the observation dictionary returned from the carla-gym environment
         @domain: the domain we used to so far"""
        domain_indicator = np.asarray([0.]) # set the indicator to zero if the domaindoes not belong the training_time available domains
        if domain in self.domain_list:
            domain_indicator = np.asarray([1.])
        ob['domain_indicator'] = domain_indicator
        return ob
    
    def initialize_agent(self, **kwargs):
        logger.debug(f"{kwargs}")
        kwargs["domain_list"] = self.domain_list
        self.agent = models.TCADT.VisionNaiveRandomization(**kwargs) # set the source domain list as one the agent's class attributes
    
    def initialize_replay_buffer():
        return None
    
    def train_module(self, module, global_step):
        logger.info(f"Training {module.__class__.__name__}...")
        info = module.fit(train_dataset=self.replay_buffer,
                          n_epochs=self.n_training_per_epoch if global_step > 0 else self.n_initial_training_epochs,
                          global_step=global_step)
        # for mode, values in info.items():
        #     self.writer.do_logging(values, global_step=global_step, mode=mode)
        
        return info
    
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
                next_ob, rew, terminated, truncated, info = self.env.step(closed_loop_action)
                self.replay_buffer.add_frame(ob, rew, terminated, truncated, info,
                                             action=expert_ac.astype(np.float32),
                                             closed_loop_action=closed_loop_action.astype(np.float32),
                                             next_state=next_ob['state'])
                fail_counter = 0

            else:
                logger.warning(f"Expert solved inaccurate with code {expert_info.get('status', 'unknown')}.")
                next_ob, rew, terminated, truncated, info = self.env.step(closed_loop_action)
                fail_counter += 1
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
    
    def sample_trajectory_with_near_future(self, domain, beta: float, pbar: Optional['tqdm'] = None,max_traj_len=np.inf,
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
                                             next_state=next_ob['state'],
                                             next_camera=next_ob['camera'].copy())
                fail_counter = 0

            else:
                logger.warning(f"Expert solved inaccurate with code {expert_info.get('status', 'unknown')}.")
                next_ob, rew, terminated, truncated, info = self.env.step(closed_loop_action)
                fail_counter += 1
                self.replay_buffer.add_frame(ob, rew, terminated, truncated, info,
                                                   action=expert_ac.astype(np.float32),
                                                   closed_loop_action=closed_loop_action.astype(np.float32),
                                                   next_state=next_ob['state'],
                                                   next_camera=next_ob['camera'].copy())
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

    def early_stop(self, evaluate_res) -> bool:
        """judge whether to early stop"""
        "find the minimum number of completed laps in all domains"
        completed_laps = min([evaluate_res[cur_map['name']]["completed_laps"] for cur_map in self.domain_list])
        return completed_laps >= 5

    def training_loop(self, n_epochs: int):

        stop_flag = False
        # Directory for saving training profiles
        profile_dir = Path(__file__).parent.parent / 'training_profiles'
        profile_dir.mkdir(parents=True, exist_ok=True)
        profile_path = profile_dir / f"{self.comment}_training_profile.pkl"
        
        for global_step in range(0, n_epochs):
            logger.info(f"Epoch {global_step} / {n_epochs}")

            cur_beta = self.init_beta * self.beta ** np.ceil(global_step / self.beta_decay_freq)
            logger.info(f"the curent beta value is {cur_beta}")
            self.sample_trajectories(beta=cur_beta,
                                        total_length=self.initial_traj_len if global_step == 0 else self.eps_len,
                                        global_step=global_step)

            train_info = self.train_module(self.agent, global_step)
        
            if global_step % self.eval_freq == 0:
                evaluate_res = self.evaluate_agent(eval_domains = self.domain_list + self.eval_domain_list)
                stop_flag = self.early_stop(evaluate_res = evaluate_res)
                
                if stop_flag and self.save_data:
                    logger.info("Convergence to successful behavior! Saving the model and the dataset ....")
                    data_dir = Path(__file__).parent.parent / 'data'
                    self.replay_buffer.export(path = data_dir, name = self.comment) # save the data only when successfully converging to successful behaviors
                    self.agent.export(path=os.path.join(Path(__file__).parent / 'model_data'), name=self.comment)
                    break
            
            self.cur_epoch = global_step
    

    def sample_trajectories(self, beta: float, total_length=None, global_step=None):
        logger.info('Sampling trajectories for training...')
        total_length = total_length or self.eps_len
        batch_traj_len = 0
        n_resets = 0
        with tqdm(total=total_length, desc='Sampling', unit='steps') as pbar:
            while batch_traj_len < total_length:

                if global_step <= 2:
                    max_traj_len = min(int(1024 / len(self.domain_list)) + 1, total_length - batch_traj_len)
                else:
                    max_traj_len = min(1024, total_length - batch_traj_len)

                sample_domain = self.sample_domain(global_step)
                traj_len = self.sample_trajectory_with_near_future(domain = sample_domain, beta = beta, pbar=pbar, max_traj_len=max_traj_len)
                batch_traj_len += traj_len
                n_resets += 1

        self.writer.do_logging({f'failure_rate': (n_resets - 1) / total_length}, global_step=global_step, mode='train')
        return batch_traj_len
    
    def sample_domain(self, global_step):
        """If the global step is the first one during training, sample domain randomly, otherwise
        sample the domain with the worst performance"""

        return np.random.choice(self.domain_list)
    
    def evaluate_agent(self, eval_domains, global_step = 0, max_laps = 7):

        logger.info("Generalization evaluations starts")

        def one_iteration_test(domain):
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

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=37) # original seed 38
    parser.add_argument('--n_epochs', type=int, default=500)
    parser.add_argument('--initial_traj_len', type=int, default=3072)
    parser.add_argument('--n_training_per_epoch', type=int, default=1)
    parser.add_argument('--n_initial_training_epochs', type=int, default=5)
    parser.add_argument('--replay_buffer_maxsize', type=int, default= 21_000) # the original replay buffer maximum size: 102_400
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
    parser.add_argument("--save_data", action = 'store_true', default = True) # whether to save the data buffer or not
    
    parser.add_argument(
        '-td', '--train_domains',
        nargs="*",
        type=str,
        default=[],
        help="List of domain names",
    )

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

    comment = params['comment']
    
    if params['evaluation']:
        params['reload'] = True # if the user wants to do evaluation, always reload the parameters

    params['comment'] = params["comment"]

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

    config_path = "./config/VisionNaiveRandomization.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    agent_params = config['model_hparams']
    trainer_cls = IL_Trainer_CARLA_VisionNaiveRandomizationAC

    try:
        train_domains = []
        for train_domain in params["train_domains"]:
            train_domains.append(domains[train_domain])
    
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
                          eval_freq=params['eval_freq'],
                          batch_size=params['batch_size'],
                          n_initial_training_epochs=params['n_initial_training_epochs'],
                          beta=params['beta'],
                          # use_labml_tracker=not params['experimental'],
                          # ntfy_freq=params['ntfy_freq'],
                          beta_decay_freq=params['beta_decay_freq'],
                          save_data = params['save_data'],
                          domain_list = train_domains,
                            **agent_params
                          )

    trainer.training_loop(n_epochs=params['n_epochs'])