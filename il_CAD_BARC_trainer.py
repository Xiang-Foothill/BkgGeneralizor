"""The trainer class for the conditional adversarial domain adaptation,
a pretrained model will be loaded together with its data_loader.
RGB images will be sampled from the target domain without aay expert supervision available.
This trainer aims for sim2real domain transfer for BARC"""

"""the example shell command for using this script:
python il_CAD_BARC_trainer.py -m "trial_02" --n_epochs 10 -c mpcc-conv -d cat_condition --seed 120"""

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
from models.adaptAC import WassersteinAdversarialAdaptAC, WassersteinConditionAdversarialAdaptAC, VisionConditionAdversarialReweightAdaptAC, VisionConditionAdversarialPseudoAdaptAC

from utils import data_util
from torch.utils.data import DataLoader
import utils.pytorch_util as ptu
from utils.logging.writer import MultiPurposeWriter
import utils.plot_util as ptl
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

PRETRAIN_SPEED0 = 'L_track_barc_speed_transfer0' # the expert is mpcc-conv, domain_list: domain4
PRETRAIN_SPEED1 = 'L_track_barc_speed_transfer1' # the expert is mpcc-conv, domain list: DOMAIN4, DOMAIN5
BARC0 = "L_track_barc_Hardware_params_model" # the expert is mpcc-conv, domain list: DOMAIN4. The expert and simulation environments are configured with time delay.
BARC1 = "L_track_barc_BARC1"
BARC2 = "L_track_barc_BARC2"
expert_mp = {
    'pid': PIDWrapper,
    'mpcc-conv': MPCCConvWrapper,
}
from il_NR_trainer import DOMAIN1, DOMAIN2, DOMAIN4, DOMAIN5, DOMAIN6, DOMAIN7, DOMAIN8, DOMAIN9, DOMAIN11, DOMAIN12, DOMAIN14, FULL_EVALUATION_LIST

class IL_Trainer_CARLA_VisionAdversarialAdaptationAC(IL_Trainer_CARLA_VisionSafeAC):

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
                       no_saving=False,
                       starting_step=0,
                       eval_freq=1,
                       batch_size=1,
                       n_initial_training_epochs=5,
                       beta=0.25,
                       pretrain_critic=False,
                       beta_decay_freq=5,
                       save_profile = False,
                       save_model = True,
                       to_reload = False,
                       target_domain_len = 5000,
                       discriminator_type = 'no_condition',
                       source_buffer = None,
                       env = None,
                       visualize = False,
                       sample_distribution = 'naive_random',
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
        self.carla_params = carla_params
        
        self.pretrain_encoder_path = BARC2 # barc 0 is the one with best sim domain performance
        self.target_domain_len = target_domain_len
        self.target_domains = [DOMAIN11]

        self.barc_data_files = [
            "ParaDriveLocalComparison_Sep7_9",
            "ParaDriveLocalComparison_Sep7_7",
            "ParaDriveLocalComparison_Oct1_enc_31",
            "ParaDriveLocalComparison_Oct1_enc_0"
        ]

        self.barc_eval_files = [
            "ParaDriveLocalComparison_Oct10_collection_0.npz"
        ]

        self.save_model = save_model
        self.visualize = visualize
        self.discriminator_type = discriminator_type
        self.sample_distribution = sample_distribution

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
        self.beta = beta
        # self.use_labml_tracker = use_labml_tracker
        self.agent_params = agent_params

        self.n_eval_success, self.n_eval_total = 0, 0
        self.eval_rewards_last10 = deque(maxlen=10)
        self.visualize_freq = 2 # the frequency of visualizing latent vectors in terms epoc num

        # an extra parameter
        self.randomnizor = linProgRandomnizer(final_percent=0.2, debug = False, mode = "constant", no_background = True) # set debug to false to speed up rendering
        transform = {"camera": self.randomnizor.traditional_randomnize}

        #initialize the data buffer
        self.replay_buffer: 'data_util.sourceTargetBalanceBuffer' = None
        self.replay_buffer = data_util.sourceTargetBalanceBuffer(maxsize=replay_buffer_maxsize,
                                        lazy_init=True,
                                        transform = transform, # when doing domain transfer, no need to have any randomization
                                        source_buffer = source_buffer)
        
        data_dir = Path(__file__).parent.parent / 'data'
        # only load the source buffer if no source buffer is given as a parameter
        if source_buffer is None:
            self.replay_buffer.source_buffer.load(path = data_dir, name = self.pretrain_encoder_path)

        if env is None:
            self.update_carla_params(carla_params)
            self.env = gym.make('barc-v0', **carla_params)
            self.eps_len = min(replay_buffer_maxsize, eps_len)
        else:
            self.env = env

        self.expert = expert_cls(dt=carla_params['dt'], t0=carla_params['t0'], track_obj=self.env.get_track())
        self.agent: 'BaseModel' = None
        self.init_beta = 1.0

        # Load previous model weights and replay buffer.
        self.starting_step = starting_step
        self.to_reload = to_reload

        if to_reload:
            
            profile_path = Path(__file__).parent.parent / 'training_profiles' / f"{self.comment}_training_profile.pkl"
            if profile_path.exists():
                with open(profile_path, 'rb') as f:
                    profile_data = pickle.load(f)
                self.init_beta = profile_data['beta']
                self.starting_step = profile_data['cur_epoch']
                self.evaluation_list = profile_data['evaluation_list']
                self.pretrain_encoder_path = profile_data['pretrain_encoder_path']
                logger.info(f"Reloaded training profile from {profile_path}")
            else:
                logger.warning(f"No training profile found at {profile_path}. Starting fresh.")

        profile_path = Path(__file__).resolve().parent / 'training_profiles' / f"{self.pretrain_encoder_path}_training_profile.pkl"
        # load the information about the pretrain agent so that initialization of the current matches the pretrained agent
        if profile_path.exists():
            with open(profile_path, 'rb') as f:
                pretrain_agent_profile_data = pickle.load(f)
            self.pretrain_agent_epochs = pretrain_agent_profile_data['cur_epoch']
            logger.info(f"the encoder has been trained for {pretrain_agent_profile_data['cur_epoch']} epochs")
            self.domain_list = pretrain_agent_profile_data["domain_list"]
            self.eval_domain_list = pretrain_agent_profile_data["eval_domain_list"]
            logger.info(f"The pretrained agent has the following domain list {[domain['name'] for domain in self.domain_list]}")
            self.pretrain_agent_model = pretrain_agent_profile_data["model_name"]

            if not to_reload:
                self.init_beta = pretrain_agent_profile_data["beta"]

                # the list used to store evaluation result
                self.evaluation_list = {}
                for domain in (self.domain_list + self.target_domains):
                    self.evaluation_list[domain["name"]] = {}
        else:
            logger.info(f"Warning: the training_profile of the pretrained_encoder is not found at {profile_path}")

        self.initialize_agent(comment=comment, ad_agent_params = agent_params)

        if to_reload:
            self.agent.load(path=Path(__file__).resolve().parent / 'model_data',
                           name=comment)

        self.save_profile = save_profile

        # Load the target domain data and the evaluation dataset
        data_path = os.path.expanduser("~/Documents/data")
        logger.info("---------- Loading target domain buffer from BARC dataset ----------")
        try:
            for barc_data_file in self.barc_data_files:
                self.replay_buffer.target_buffer.load_barc_data(data_path = data_path, data_name = barc_data_file)
        finally:
            logger.info("------- Barc data loading finished ---------")
        
        logger.info("----- Generating barc evaluation buffer -----")
        self.eval_buffer = data_util.EfficientReplayBuffer(maxsize = 5000, lazy_init=True, transform = None)
        for barc_data_file in self.barc_eval_files:
            self.eval_buffer.load_barc_data(data_path = data_path, data_name = barc_data_file)
        logger.info("----- Evaluation buffer generated -----")

        self.writer: 'MultiPurposeWriter' = MultiPurposeWriter(model_name=self.agent.model_name,
                                                               log_dir=f"logs/{self.agent.model_name}_{comment or ''}",
                                                               comment=comment or '',
                                                               print_method=logger.info,
                                                               # use_labml_tracker=use_labml_tracker,
                                                               # ntfy_freq=ntfy_freq,
                                                               )
            
    def initialize_agent(self, comment, ad_agent_params):
        pretrain_config_path = "./config/VisionAttentionActor.yaml"
        with open(pretrain_config_path, 'r') as f:
            config = yaml.safe_load(f)

        pretrain_agent_params = config['model_hparams']

        # modify the learning rate of the pretrained agent for better fine tuning
        pretrain_agent_params['lr'] = ad_agent_params['lr_actor']
        pretrain_agent = VisionNaiveRandomization(**pretrain_agent_params)

        logger.info("//// Loading pretrained encoders ////")
        pretrain_agent.load(path=Path(__file__).resolve().parent / 'model_data',
                           name=self.pretrain_encoder_path)

        actor_cls = VisionAdversarialAdaptAC
        if self.discriminator_type == 'cat_condition':
            actor_cls = VisionConditionAdversarialAdaptAC
        if self.discriminator_type == 'null':
            actor_cls = VisionNullAdaptAC
        if self.discriminator_type == 'proj_condition':
            actor_cls = VisionProjConditionAdversarialAC
        if self.discriminator_type == 'cat_condition_reweight':
            actor_cls = VisionConditionAdversarialReweightAdaptAC
            ad_agent_params['sample_distribution'] = self.sample_distribution # pass in the sample distribution of the best bandwidth initialization
            ad_agent_params['pretrain_name'] = self.pretrain_encoder_path
        if self.discriminator_type == 'cat_condition_pseudo':
            actor_cls = VisionConditionAdversarialPseudoAdaptAC
            ad_agent_params['target_buffer_max_size'] = 1024
            
        self.agent = actor_cls(pretrain_agent = pretrain_agent, pretrain_agent_params = pretrain_agent_params, ad_agent_params = ad_agent_params)
        logger.debug(f"the loaded model is {self.agent.model_name}")

    def initialize_replay_buffer():
        return None
    
    def train_module(self, module : VisionAdversarialAdaptAC, global_step):
        logger.info(f"Training {module.__class__.__name__}...")
        info = module.fit(train_dataset=self.replay_buffer, val_dataset = self.eval_buffer, # set the eval_buffer to evaluate the agent's performance on the real domain
                          n_epochs= self.n_training_per_epoch if global_step > 0 else self.n_initial_training_epochs,
                          global_step=global_step)
        
        dis_info, actor_info = info
        # hard-code the quantities that we care about
        self.writer.add_scalars(main_tag="policy MSE", tag_scalar_dict = {"sim_domain": actor_info['train']["policy_loss"], "real_domain" : actor_info["val"]["policy_loss"]}, global_step = global_step)
        self.writer.add_scalar("train_time adversarial loss", scalar_value = actor_info['train']["adversarial_loss"], global_step = global_step)
        self.writer.add_scalar("train_time disriminator loss", scalar_value = dis_info['train']["discriminator_loss"], global_step = global_step)
    
    def label_domain(self, ob, domain):
        """add the field 'domain_v', the domain probability vector to the ob
         @ob: the observation dictionary returned from the carla-gym environment
         @domain: the domain we used to so far"""
        domain_indicator = np.asarray([0.]) # set the indicator to zero if the domaindoes not belong the training_time available domains
        if domain in self.domain_list:
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
                buffer.add_frame(ob, rew, terminated, truncated, info,
                                             action=expert_ac.astype(np.float32),
                                             closed_loop_action=closed_loop_action.astype(np.float32),
                                             next_state=next_ob['state'])
                fail_counter = 0

            else:
                logger.warning(f"Expert solved inaccurate with code {expert_info.get('status', 'unknown')}.")
                next_ob, rew, terminated, truncated, info = self.env.step(closed_loop_action)
                fail_counter += 1
                # do not add poorly labeled results to the dataset
                # buffer.add_frame(ob, rew, terminated, truncated, info,
                #                                    action=expert_ac.astype(np.float32),
                #                                    closed_loop_action=closed_loop_action.astype(np.float32),
                #                                    next_state=next_ob['state'])
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

    def visualize_action_diff(self, buffer, *, max_points=None, sample_stride=1,
                            cmap='gray_r', robust=True, s=14, alpha=0.95):
        """
        Visualize ||agent_action - expert_action|| at each GPS (x,y) on the track.

        Parameters
        ----------
        buffer : EfficientReplayBuffer or sequence-like
            Each item supports:
            - item["gps"]    -> array-like, first two entries are (x, y)
            - item["action"] -> expert action, shape (..., 2) or (2,)
        max_points : int | None
            If set, cap the number of samples visualized (after striding).
        sample_stride : int
            Take every `sample_stride`-th item from the buffer (default 1).
        cmap : str
            Colormap; 'gray_r' makes darker = larger difference.
        robust : bool
            If True, use robust normalization (2–98 percentile).
        s, alpha : scatter styling
        """
        # ----- gather data -----
        N_total = len(buffer)
        idxs = np.arange(0, N_total, sample_stride)
        if max_points is not None:
            idxs = idxs[:max_points]

        gps_list, agent_list, expert_list = [], [], []

        for i in idxs:
            item = buffer[i]  # assumes __getitem__ returns a dict-like sample

            # gps (x,y)
            gps_xy = np.asarray(item["gps"][:2], dtype=float)
            if not np.all(np.isfinite(gps_xy)):
                continue

            # expert action (first 2 dims)
            exp_act = np.asarray(item["action"], dtype=float).reshape(-1)
            if exp_act.size < 2 or not np.all(np.isfinite(exp_act[:2])):
                continue
            exp_act = exp_act[:2]

            # agent action via your policy API
            obs_args = self.agent.parse_carla_obs(obs=item, info={})
            ag_act, _ = self.agent.get_action(*obs_args)
            ag_act = np.asarray(ag_act, dtype=float).reshape(-1)
            if ag_act.size < 2 or not np.all(np.isfinite(ag_act[:2])):
                continue
            ag_act = ag_act[:2]

            gps_list.append(gps_xy)
            expert_list.append(exp_act)
            agent_list.append(ag_act)

        if len(gps_list) == 0:
            raise ValueError("No valid (gps, agent_action, expert_action) triplets found to plot.")

        data = {
            "gps": np.vstack(gps_list),                 # (M, 2)
            "agent_action": np.vstack(agent_list),      # (M, 2)
            "expert_action": np.vstack(expert_list),    # (M, 2)
        }

        # ----- plot on track -----
        fig, ax = plt.subplots(figsize=(10, 8))
        # support either get_trak() or get_track()
        get_trak = getattr(self.env, "get_trak", None)
        track_obj = get_trak() if callable(get_trak) else self.env.get_track()
        track_obj.plot_map(ax)
        ax.set_aspect('equal')
        ax.set_title('Action Discrepancy (Agent vs Expert) on Track')

        # use your existing helper
        ptl.plot_action_diff_on_track(data, ax, cmap=cmap, robust=robust, s=s, alpha=alpha, add_colorbar=True)

        plt.tight_layout()
        plt.show()
        return fig, ax


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

    def sample_trajectory_with_future(self, domain, beta: float, pbar: Optional['tqdm'] = None,
                          max_traj_len=np.inf,
                          PATIENCE=2, TRUNCATE=np.inf, buffer: data_util.EfficientReplayBuffer = None):
        
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
            self.label_domain(ob, domain)
            ac, _ = self.agent.get_action(*self.agent.parse_carla_obs(ob, info))
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
            buffer.add_frame(
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
    
    def training_loop(self, n_epochs: int):
        max_traj_len = 0

        def make_stop_flag(consecutive_steps = 1, success_laps = 4):
            """early stop mechnism, if we have consecutive evaluations with 
            completed laps larger than success_laps, the experiment is called to early stop,
            i.e. the system is recognized to converge to stable behavior"""
            def f_stop_flag(evaluate_res):
                return False
            return False, f_stop_flag
            
        stop_flag, f_stop_flag = make_stop_flag()
        visualization_list = [self.domain_list[0], self.target_domains[0]]
        
        # before starting tuning the model for domain adpatation, do the evaluation for different domains
        logger.info("Pretraining Evaluation .......")
        if self.visualize:
            self.PCA_visualization()

        # Directory for saving training profiles
        profile_dir = Path(__file__).parent.parent / 'training_profiles'
        profile_dir.mkdir(parents=True, exist_ok=True)
        profile_path = profile_dir / f"{self.comment}_training_profile.pkl"
            
        """start the agent domain adaptation"""
        logger.info("--------------Adversarial Domain Adaptation starts-------------")

        for global_step in range(self.starting_step, n_epochs):
            # self.visualize_action_diff(buffer = self.replay_buffer.target_buffer, max_points = 1000)
            # ptl.visualize_random_rgb_with_actions(dataloader=self.replay_buffer.source_buffer)

            logger.info(f"Epoch {global_step} / {n_epochs} for the decision layer [Epoch {global_step + self.pretrain_agent_epochs} / {n_epochs + self.pretrain_agent_epochs} for the whole model]")
            
            self.randomnizor.update_cur(global_step = global_step, total_epochs=n_epochs)
            self.train_module(self.agent, global_step)

            cur_beta = 0.3 * (self.beta ** (global_step - self.starting_step)) # make the data collection process more stable
            logger.info(f"the current beta is {cur_beta}")
            self.sample_trajectories(beta=cur_beta,
                                     domain_list = self.domain_list,
                                    total_length=self.eps_len,
                                    buffer = self.replay_buffer.source_buffer,
                                    global_step=global_step)

            if global_step % self.visualize_freq == 0 and self.visualize:
                self.PCA_visualization()

            self.cur_epoch = global_step

        if self.save_model:
            self.agent.export(path=os.path.join(Path(__file__).parent / 'model_data'), name=self.comment)

    def PCA_visualization(self, max_per_domain=150):
        """
        Visualize latent vectors from the RGB images already stored in replay buffers.
        Reads:
            - self.replay_buffer.source_buffer[i]["camera"]
            - self.replay_buffer.target_buffer[i]["camera"]
        Only argument: max_per_domain — maximum number of samples per domain.
        """

        # === Helper functions ===
        def _len(buf):
            try:
                return len(buf)
            except Exception:
                return 0

        def preprocess_images(img_list):
            # Return raw uint8 -> let get_latent do the /255
            return np.stack([np.asarray(img, dtype=np.uint8) for img in img_list], axis=0)

        def get_latents(buf, n_samples):
            """Randomly sample up to n_samples images from buffer and encode."""
            if buf is None or len(buf) == 0:
                return np.empty((0,))
            idxs = np.random.choice(len(buf), size=min(n_samples, len(buf)), replace=False)
            imgs = [buf[i]["camera"] for i in idxs]
            imgs = preprocess_images(imgs)
            imgs_t = ptu.from_numpy(imgs.copy())
            with torch.no_grad():
                latents = self.agent.get_latent(imgs_t)
                if torch.is_tensor(latents):
                    latents = latents.detach().cpu().numpy()
            return latents

        # === Get buffers ===
        src_buf = getattr(self.replay_buffer, "source_buffer", None)
        tgt_buf = getattr(self.replay_buffer, "target_buffer", None)
        n_src, n_tgt = _len(src_buf), _len(tgt_buf)

        if n_src == 0 and n_tgt == 0:
            logger.error("Both source_buffer and target_buffer are empty or missing.")
            return

        # === Encode latents ===
        logger.info("Encoding latent vectors from stored RGBs...")
        src_lat = get_latents(src_buf, max_per_domain) if n_src > 0 else np.empty((0,))
        tgt_lat = get_latents(tgt_buf, max_per_domain) if n_tgt > 0 else np.empty((0,))

        parts, labels, names = [], [], []
        if src_lat.size > 0:
            parts.append(src_lat)
            labels.append(np.zeros(len(src_lat), dtype=int))
            names.append("source domain")
        if tgt_lat.size > 0:
            parts.append(tgt_lat)
            labels.append(np.ones(len(tgt_lat), dtype=int))
            names.append("target domain")

        if not parts:
            logger.warning("No latent vectors produced. Check replay buffers.")
            return

        X = np.vstack(parts)
        y = np.concatenate(labels)

        # === PCA ===
        logger.info("Applying PCA projection...")
        pca = PCA(n_components=2, random_state=0)
        X2 = pca.fit_transform(X)
        evr = pca.explained_variance_ratio_

        # === Plot ===
        plt.figure(figsize=(8, 6))
        colors = ["tab:blue", "tab:orange"]
        for i, name in enumerate(names):
            mask = (y == i)
            plt.scatter(X2[mask, 0], X2[mask, 1], s=6, alpha=0.6, label=name, c=colors[i])

        plt.title("PCA of Latent Vectors (from replay buffers)")
        plt.xlabel(f"PC1 ({evr[0]*100:.1f}% var)")
        plt.ylabel(f"PC2 ({evr[1]*100:.1f}% var)")
        plt.legend()
        plt.grid(True, linewidth=0.3)
        plt.tight_layout()
        plt.show()

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

        eval_domain = DOMAIN11
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

    def main(self, n_epochs: int):
        self.training_loop(n_epochs=n_epochs)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=37) # original seed 38
    parser.add_argument('--n_epochs', type=int, default=500)
    parser.add_argument('--initial_traj_len', type=int, default=3072)
    parser.add_argument('--n_training_per_epoch', type=int, default=1)
    parser.add_argument('--n_initial_training_epochs', type=int, default=1) 
    parser.add_argument('--replay_buffer_maxsize', type=int, default= 51_200) # the original replay buffer maximum size: 102_400
    parser.add_argument('--expert', '-c', type=str, default='mpcc-conv',
                        choices=tuple(expert_mp.keys()))
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--comment', '-m', type=str, default='')
    parser.add_argument('--eps_len', type=int, default=64)
    parser.add_argument('--randomnize', default = '', choices = ('', 'pure_augment', "random_fetch", "contrast"))

    parser.add_argument('--town', type=str, default='L_track_barc')
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--no_saving', action='store_true')
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--beta', type=float, default=0.8)
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
    # parser.add_argument('--ntfy_freq', type=int, default=100)
    parser.add_argument("--target_domain_len", '-t', type = int, default = 2048) # the length of total trajectory sampled from the target domain
    parser.add_argument("--discriminator", '-d', type = str, default = 'no_condition', choices = ('no_condition', 'cat_condition', 'proj_condition','null', 'cat_condition_reweight', 'cat_condition_pseudo'))
    parser.add_argument("--sample_distribution", '-s', type = str, default = 'naive_random', choices = ('naive_random', 'first_4m_random', 'middle_3m_random'))

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

    params['comment'] = '_'.join((params['town'], params['comment']))

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

    d_type = params['discriminator']

    config_path = "./config/VisionAD.yaml" if (d_type == 'no_condition' or d_type == 'null') else "./config/VisionCAD.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    agent_params = config['model_hparams']
    trainer_cls = IL_Trainer_CARLA_VisionAdversarialAdaptationAC

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
                          target_domain_len = params["target_domain_len"],
                          discriminator_type = params['discriminator'],
                          sample_distribution = params['sample_distribution'],
                          **agent_params
                          )

    if params['evaluation']:
        logger.info(f"the evaluated agent has been trained for {trainer.starting_step + trainer.pretrain_agent_epochs} epochs")
        trainer.evaluate_agent(eval_domains = FULL_EVALUATION_LIST, global_step=0)

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