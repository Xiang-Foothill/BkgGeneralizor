import itertools
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torchvision.models import resnet18, ResNet18_Weights

from utils import pytorch_util as ptu
from .base_model import BaseModel
from .safeAC import Dynamics, SafeCritic, SafeAC
import pathlib as Path
from loguru import logger
from utils.data_util import EfficientReplayBuffer, EfficientReplayBufferPN, sourceTargetBalanceBuffer
import torch.nn.functional as F
import copy
from tqdm import tqdm
import math
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import os, yaml
from typing import List, Callable
from itertools import product

class VisionNaiveRandomization(BaseModel):
    feature_fields = ['camera', 'velocity']
    label_fields = ['action', 'state']

    def __init__(self, ob_dim, ac_dim, size, n_layers, lr=1e-3, weight_decay=1e-5, lam=1., l_dim = 10, gamma = 0.977, domain_list : List = []):
        """
        Model Input: states
        Model Output: actions
        domain_list: the list of source domains

        Loss: MSE
        """
        super().__init__()
        self.initialize_encoders(ob_dim)
        self.domain_list = domain_list
        
        self.decision = nn.Linear(512 + 16, ac_dim)
        logger.info(f"The learning rate = {lr}")

        self.optimizer = Adam(itertools.chain(self.resnet.parameters(), self.decision.parameters(), self.velocity_encoder.parameters()), lr=lr,
                              weight_decay=weight_decay)
        self.scheduler = ExponentialLR(optimizer=self.optimizer, gamma=gamma)  # 0.1 ** (1 / 100))
        self.loss_func = nn.MSELoss()
        self.log_sigmoid = nn.LogSigmoid()
        self.lam = lam
        
        self._extra_attrs_to_save = ["domain_list"]
    
    def initialize_encoders(self, ob_dim):
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Sequential(
        )
        self.velocity_encoder = nn.Linear(ob_dim, 16)

    def step_schedule(self):
        self.lam = min(10., self.lam * 1.007)  # Approx 2 ** (1 / 100)
    
    def loss(self, pred, label, k=1, temperature=1):
        u_pred, = pred # the u_pred here has a shape of [64, 2], [batch_size, output_size]
        u, states = label
        mse_loss = self.loss_func(u_pred, u)
        return mse_loss, {'mse_loss': mse_loss.item()}
    
    def forward(self, img, vel):
        # Normalize the image first. 
        img = img.permute(0, 3, 1, 2) / 255.
        # img = (img / 255. - self.img_mean) / self.img_std
        # logger.debug(f"{img.size()}")
        l = self.resnet(img)

        if self.check_latent_collapse(l):
            logger.info("WARNING: latent space collapse is detected!")

        v_encoded = self.velocity_encoder(vel)
        out = self.decision(torch.cat([l, v_encoded], dim=1))
        return (out,)
    
    def check_latent_collapse(self, l, threshold=1e-5):
        """check whether the latent space collapse occurs, i.e. all the latent vectors fall into the same point"""

        """
        Check whether latent space collapse occurs, without affecting gradient computation.

        Args:
        l (torch.Tensor): A batch of latent vectors of shape (batch_size, latent_dim)
        threshold (float): Minimum allowed variance before considering collapse

        Returns:
        bool: True if latent collapse detected, False otherwise
        """

        if l.shape[0] == 1:
            return False
        
        with torch.no_grad():
            variance = l.var(dim=0)
            collapsed = torch.all(variance < threshold).item()
        return collapsed

    def freeze_encoders(self):
        """this method is called during the middle of the training time to
        freeze the visual encoder and the velocity encoder at the same time,
        the only part of the model that is left to be tuned is the decision layer"""

        logger.info("////// Middle Freezing Triggered ////// [the visual encoder and the velocity encoder is no longer tunable]")
        # Freeze ResNet (visual encoder)
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Freeze velocity encoder
        for param in self.velocity_encoder.parameters():
            param.requires_grad = False

        # (Re)initialize optimizer to only update decision layer
        self.optimizer = Adam(
            self.decision.parameters(),
            lr=self.optimizer.param_groups[0]['lr'],
            weight_decay=self.optimizer.param_groups[0]['weight_decay']
        )
    
    def get_latent(self, img):
        self.eval()
        img = img.permute(0, 3, 1, 2) / 255.
        l = self.resnet(img)

        return ptu.to_numpy(l)

class VisionAdversarialActor(BaseModel):
    """Unlike VisionAdversarialAdaptActor_modes, this adversarial actor is not implemented based on the mode mechanism"""
    feature_fields = ['camera', 'velocity']
    label_fields = ['action', "domain_indicator"]

    def __init__(self, pretrain_agent : VisionNaiveRandomization, adv_factor = 0.75):
        """
        Model Input: states
        Model Output: actions

        Loss: MSE
        """
        super().__init__()
        self.resnet = pretrain_agent.resnet
        self.decision = pretrain_agent.decision
        self.velocity_encoder = pretrain_agent.velocity_encoder
        self.optimizer = pretrain_agent.optimizer
        self.scheduler = pretrain_agent.scheduler
        self.loss_func = nn.MSELoss()

        self.adv_factor = adv_factor # the factor multiplied by the adversarial loss before adding it to the total loss
    
    def freeze(self):
        logger.info(f"freeze the {self.model_name}")
        for p in self.resnet.parameters():
            p.requires_grad = False
        for p in self.decision.parameters():
            p.requires_grad = False
        for p in self.velocity_encoder.parameters():
            p.requires_grad = False
    
    def unfreeze(self):
        logger.info(f"unfreeze the {self.model_name}")
        for p in self.resnet.parameters():
            p.requires_grad = True
        for p in self.decision.parameters():
            p.requires_grad = True
        for p in self.velocity_encoder.parameters():
            p.requires_grad = True

    def set_discriminator(self, discriminator):
        self.discriminator = discriminator
    
    def get_latent(self, img, to_numpy = True):

        img = img.permute(0, 3, 1, 2) / 255.
        l = self.resnet(img)

        if to_numpy: # convert the latent vectors to numpy array
            return ptu.to_numpy(l)
        else:
            return l
    
    def forward(self, img, vel):

        # Normalize the image first. 
        img = img.permute(0, 3, 1, 2) / 255.

        l = self.resnet(img)
        domain_logits, = self.discriminator(l)  # allow gradients to flow

        if self.check_latent_collapse(l):
            logger.info("WARNING: latent space collapse is detected!")

        v_encoded = self.velocity_encoder(vel)
        combined = torch.cat([l, v_encoded], dim=1)  # shape: [batch_size, latent_dim]

        output = self.decision(combined)
        return (output, domain_logits)

    def loss(self, pred, label, only_source_policy = True):
        """only_source_policy: bool. If it is true, the MSE policy loss will only be calculated using source-domain examples. Otherwise
        all examples will be used to calculate MSE loss."""

        u_pred, domain_logits = pred # the u_pred here has a shape of [64, 2], [batch_size, output_size]
        u, domain_ind = label

        domain_ind = domain_ind.view(-1, 1)  # Ensure consistent shape: [batch_size, 1]
        #    === Source mask for policy loss ===
        source_mask = domain_ind.bool().squeeze(1)  # shape: [batch_size]

        # === Policy loss: only for source samples ===
        if only_source_policy:
            if source_mask.sum() > 0:
                policy_loss = self.loss_func(u_pred[source_mask], u[source_mask])
            else:
                policy_loss = torch.tensor(0.0, device=u.device)
        else:
            policy_loss = self.loss_func(u_pred, u)

        # Adversarial loss: encourage encoder to fool discriminator
        # Flip domain labels: try to make domain_ind_pred look like the opposite (e.g., 0.5 or source)
        # target_labels = 1. - domain_ind.float()  # invert labels
        target_labels = torch.ones_like(domain_ind, dtype=torch.float32) * 0.5  # new uniform label: 0.9 for all, which is the best label for PRETRAIN_BRIGHT3

        adv_loss_fn = nn.BCEWithLogitsLoss()
        adv_loss = adv_loss_fn(domain_logits, target_labels)
        total_loss = policy_loss + self.adv_factor * adv_loss

        return total_loss, {
                'policy_loss': policy_loss.item(),
                'adversarial_loss': adv_loss.item(),
                'total_loss': total_loss.item()
        }
    
    def fit(self, train_dataset: sourceTargetBalanceBuffer, n_epochs, val_dataset=None, global_step = None):
        train_examples, train_loss = 0, 0.
        val_examples, val_loss = 0, 0.

        train_loader = train_dataset.series_dataloader(batch_size=64, shuffle=True, num_workers=0,
                                                manifest=[self.feature_fields, self.label_fields])
        val_loader = val_dataset.dataloader(batch_size=64, shuffle=False, num_workers=0,
                                            manifest=[self.feature_fields,
                                                      self.label_fields]) if val_dataset is not None else None
        train_scores = defaultdict(lambda: 0.)
        val_scores = defaultdict(lambda: 0.) if val_dataset is not None else None

        self.discriminator.freeze()

        for epoch in range(n_epochs):
            self.train()
            for features, labels in tqdm(train_loader, desc = f'[epoch {epoch}]'):
                pred = self(*features)
                loss, train_info = self.loss(pred, labels, only_source_policy=True)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                self.optimizer.step()
                train_loss += loss.item()
                train_examples += 1
                for k, v in train_info.items():
                    train_scores[k] += v

            if val_loader is None:
                continue

            self.eval()
            with torch.no_grad():
                for features, labels in val_loader:
                    pred = self(*features)
                    loss, val_info = self.loss(pred, labels, only_source_policy=False)
                    val_loss += loss.item()
                    val_examples += 1
                    for k, v in val_info.items():
                        val_scores[k] += v
        self.scheduler.step()
        info = {
            'train': {f'loss': train_loss / train_examples,
                      **{f'{k}': v / train_examples for k, v in train_scores.items()}}
        }

        for k, v in train_scores.items():
            logger.info(f"train-time {k} = {v / train_examples}")

        if val_loader is not None:
            info['val'] = {f'loss': val_loss / val_examples,
                           **{f'{k}': v / val_examples for k, v in val_scores.items()}}
        if val_dataset is not None:
            for k, v in val_scores.items():
                logger.info(f"eval-time real-domain {k} = {v / val_examples}")
        
        self.discriminator.unfreeze()

        return info
    
    def get_action(self, *args):
        self.eval()
        outputs = [x for x in self(*[ptu.from_numpy(arg.copy()[None]) for arg in args])]
        for i, output in enumerate(outputs):
            outputs[i] = ptu.to_numpy(output)
        u_pred, domain_label = outputs[0][0], outputs[1][0]
        return u_pred, domain_label
    
    def check_latent_collapse(self, l, threshold=1e-5):
        """check whether the latent space collapse occurs, i.e. all the latent vectors fall into the same point"""

        """
        Check whether latent space collapse occurs, without affecting gradient computation.

        Args:
        l (torch.Tensor): A batch of latent vectors of shape (batch_size, latent_dim)
        threshold (float): Minimum allowed variance before considering collapse

        Returns:
        bool: True if latent collapse detected, False otherwise
        """

        if l.shape[0] == 1:
            return False
        
        with torch.no_grad():
            variance = l.var(dim=0)
            collapsed = torch.all(variance < threshold).item()
        return collapsed
    
    def parse_obs(self, vehicle_state, env_state):
        """parse the observation from BARC"""
        return (
            np.moveaxis(env_state[0], [0, 1, 2], [1, 2, 0]), # make the image channel last
            np.array([vehicle_state.v.v_long, vehicle_state.v.v_tran, vehicle_state.w.w_psi]) # get velocity vectors, note that in our problem setting, v_tran is part of v
        )

class Discriminator(BaseModel):

    """the discriminator for domain adversarial transfer, this is the most basic version of domain adversarial transfer
    No conditioning policy is applied"""

    feature_fields = ['camera']
    label_fields = ['domain_indicator']

    def __init__(self, lr, weight_decay, encoder_output_dim, dis_info_mode = "None", null_init = False):
        super().__init__()

        if null_init:
            return

        #define dis_info_dim based on the dis_info_mode
        if dis_info_mode == "only_curvature" or dis_info_mode == "gps_full":
            dis_info_dim = 3
        elif dis_info_mode == "state_curvature":
            dis_info_dim = 5
        elif dis_info_mode == "only_state" or dis_info_mode == "gps_xy":
            dis_info_dim = 2
        elif dis_info_mode == "only_x_tran":
            dis_info_dim = 1
        elif dis_info_mode == "None":
            dis_info_dim = 0
        else:
            raise ValueError("The input dis_info_mode is invalid")
        self.dis_info_mode = dis_info_mode
        logger.info(f"The current applied discriminative information is {self.dis_info_mode}")

        #TODO: Make the discriminator deeper to improve its experssive ability
        self.D = nn.Sequential(
            # the first layer block #
            nn.Linear(encoder_output_dim + dis_info_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(p=0.1), # add dropouts and layerNorm for the purpose of stable training

            # the second layer block #
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(p = 0.1),

            nn.Linear(128, 1)
        )
        # such a discriminator by default set the input size to be 512

        # self.D = nn.Linear(encoder_output_dim + dis_info_dim, 1) # the simplest version of discrminator network

        self.optimizer = Adam(self.D.parameters(), lr=lr,
                              weight_decay=weight_decay)
        self.scheduler = ExponentialLR(optimizer=self.optimizer, gamma=1.0) # no decay at all time
    
    def freeze(self):
        logger.info(f"freeze the {self.model_name}")
        for p in self.D.parameters():
            p.requires_grad = False
    
    def unfreeze(self):
        logger.info(f"unfreeze the {self.model_name}")
        for p in self.D.parameters():
            p.requires_grad = True
    
    def input_buffer(self, features, actor : VisionAdversarialActor ):
        """preprocess the features before passing them to the model.
        note that in the dataloader, there are only RGB images, there is no such a field called latent vector"""
        latent_vectors = actor.get_latent(img = features[0], to_numpy = False)
        latent_vectors = latent_vectors.detach() # cut from the rest of the computation graph
        return [latent_vectors]

    def fit(self, train_dataset: sourceTargetBalanceBuffer, n_epochs, val_dataset : sourceTargetBalanceBuffer = None, global_step = None, actor : VisionAdversarialActor = None):
        train_examples, train_loss = 0, 0.
        val_examples, val_loss = 0, 0.

        train_loader = train_dataset.balanced_dataloader(batch_size=64, shuffle=True, num_workers=0,
                                                manifest=[self.feature_fields, self.label_fields])
        val_loader = val_dataset.balanced_dataloader(batch_size=64, shuffle=False, num_workers=0,
                                            manifest=[self.feature_fields,
                                                      self.label_fields]) if val_dataset is not None else None
        train_scores = defaultdict(lambda: 0.)
        val_scores = defaultdict(lambda: 0.) if val_dataset is not None else None
        actor.freeze()

        for epoch in range(n_epochs):
            self.train()
            for features, labels in tqdm(train_loader, desc = f'[epoch {epoch}]'):
                #TO DO: add augmentation functions here to the feature class
                latent_features = self.input_buffer(features, actor = actor)
                pred = self(*latent_features)
                loss, train_info = self.loss(pred, labels)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                self.optimizer.step()
                train_loss += loss.item()
                train_examples += 1
                for k, v in train_info.items():
                    train_scores[k] += v

            if val_loader is None:
                continue

            self.eval()
            actor.eval()

            with torch.no_grad():
                for features, labels in val_loader:
                    latent_features = self.input_buffer(features, actor = actor)
                    pred = self(*latent_features)
                    loss, val_info = self.loss(pred, labels)
                    val_loss += loss.item()
                    val_examples += 1
                    for k, v in val_info.items():
                        val_scores[k] += v
            
            # val_dis_loss = val_scores['discriminator_loss'] / val_examples # the validation discrimiantor loss

        self.scheduler.step()
        info = {
            'train': {f'loss': train_loss / train_examples,
                      **{f'{k}': v / train_examples for k, v in train_scores.items()}}
        }

        for k, v in train_scores.items():
            logger.info(f"{k} = {v / train_examples}")

        if val_loader is not None:
            info['val'] = {f'loss': val_loss / val_examples,
                           **{f'{k}': v / val_examples for k, v in val_scores.items()}}
        
        actor.unfreeze()

        return info
    
    def fit_ES(self, 
           train_dataset: sourceTargetBalanceBuffer, 
           n_epochs,  # MAX number of fitting epochs
           val_dataset: sourceTargetBalanceBuffer = None, 
           global_step=None, 
           actor: VisionAdversarialActor = None,
           debug: bool = False):
        """
        Early-stopping (Option 2.B) using EMA of *per-epoch* val_dis_loss.
        - n_epochs is a hard cap (max epochs).
        - Early stop when EMA hasn't improved by >= eps for `patience_up` validations.
        - Early-stopping is based on the mean val_dis_loss over ONLY the batches of the current epoch.
        - If debug=True, plot per-epoch curves for:
            * val_dis_loss_epoch
            * val_dis_loss_ema
            * train_loss_epoch (mean over batches this epoch)
        - All other behavior (accumulators, logging, return format) is unchanged.
        """

        # ---- ES hyperparams ----
        eps = 5e-2
        patience_up = 2
        ema_beta = 0.1  # smoothing factor for EMA over per-epoch val_dis_loss

        # ---- Optional plotting setup ----
        if debug:
            import matplotlib.pyplot as plt
            dbg_epochs = []
            dbg_val_epoch = []
            dbg_val_ema = []
            dbg_train_epoch = []

        # ---- Original accumulators (unchanged for final info) ----
        train_examples, train_loss = 0, 0.0
        val_examples, val_loss = 0, 0.0
        from collections import defaultdict
        train_scores = defaultdict(lambda: 0.0)
        val_scores = defaultdict(lambda: 0.0) if val_dataset is not None else None

        train_loader = train_dataset.balanced_dataloader(
            batch_size=64, shuffle=True, num_workers=0,
            manifest=[self.feature_fields, self.label_fields]
        )
        val_loader = val_dataset.balanced_dataloader(
                batch_size=64, shuffle=True, num_workers=0,
                manifest=[self.feature_fields, self.label_fields]
            ) if val_dataset is not None else None

        actor.freeze()

        # ---- ES state ----
        val_dis_loss_ema = None
        best_ema = None
        no_improve = 0
        early_stop = False

        for epoch in range(n_epochs):  # n_epochs is a cap
            self.train()

            # Per-epoch accumulators (for debug plotting only)
            train_loss_epoch, train_examples_epoch = 0.0, 0

            for features, labels in tqdm(train_loader, desc=f'[epoch {epoch}]'):
                latent_features = self.input_buffer(features, actor=actor)
                pred = self(*latent_features)
                loss, train_info = self.loss(pred, labels)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                self.optimizer.step()

                # Global accumulators (unchanged)
                train_loss += loss.item()
                train_examples += 1
                for k, v in train_info.items():
                    train_scores[k] += v

                # Per-epoch (for debug)
                train_loss_epoch += loss.item()
                train_examples_epoch += 1

            # ---- VALIDATION ----
            val_dis_loss_epoch = None  # default if no val or metric not present
            if val_loader is not None:
                # Per-epoch accumulators for ES decision
                val_examples_epoch = 0
                val_scores_epoch = defaultdict(lambda: 0.0)

                self.eval()
                with torch.no_grad():
                    for features, labels in val_loader:
                        latent_features = self.input_buffer(features, actor=actor)
                        pred = self(*latent_features)
                        loss, val_info = self.loss(pred, labels)

                        # Global (unchanged) accumulators for final info
                        val_loss += loss.item()
                        val_examples += 1
                        for k, v in val_info.items():
                            val_scores[k] += v

                        # Per-epoch accumulators (ES decision)
                        val_examples_epoch += 1
                        for k, v in val_info.items():
                            val_scores_epoch[k] += v

                # ---- Compute per-epoch val_dis_loss (NEW) ----
                if 'discriminator_loss' in val_scores_epoch and val_examples_epoch > 0:
                    val_dis_loss_epoch = val_scores_epoch['discriminator_loss'] / val_examples_epoch

                    # EMA smoothing over per-epoch values
                    if val_dis_loss_ema is None:
                        val_dis_loss_ema = val_dis_loss_epoch
                        best_ema = val_dis_loss_ema
                        no_improve = 0
                    else:
                        val_dis_loss_ema = ema_beta * val_dis_loss_ema + (1 - ema_beta) * val_dis_loss_epoch
                        if val_dis_loss_ema < best_ema - eps:
                            best_ema = val_dis_loss_ema
                            no_improve = 0
                        else:
                            no_improve += 1

                    # ---- Early-stop condition (Option 2.B) ----
                    if no_improve >= patience_up:
                        # No logger prints per your request
                        early_stop = True

            # ---- Debug plotting ----
            if debug:
                # Compute this epoch's average training loss (safe division)
                train_loss_epoch_mean = (train_loss_epoch / train_examples_epoch) if train_examples_epoch else float('nan')
                dbg_epochs.append(epoch)
                dbg_train_epoch.append(train_loss_epoch_mean)
                # Only append val metrics if computed this epoch
                if val_dis_loss_epoch is not None:
                    dbg_val_epoch.append(val_dis_loss_epoch)
                    dbg_val_ema.append(val_dis_loss_ema)
                else:
                    dbg_val_epoch.append(float('nan'))
                    dbg_val_ema.append(float('nan'))

            if early_stop:
                break
        if debug:
            # Draw/update a simple 1-figure, 3-series plot
            plt.figure(figsize=(6, 4))
            plt.title("Discriminator ES Debug (per-epoch)")
            plt.plot(dbg_epochs, dbg_val_epoch, marker='o', label='val_dis_loss_epoch')
            plt.plot(dbg_epochs, dbg_val_ema, marker='o', label='val_dis_loss_ema')
            plt.plot(dbg_epochs, dbg_train_epoch, marker='o', label='train_loss_epoch')
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.legend()
            plt.tight_layout()
            plt.show()

        # Keep scheduler placement unchanged (after loop)
        self.scheduler.step()

        # ---- Return info (unchanged format; still cumulative means) ----
        info = {
            'train': {
                'loss': train_loss / train_examples if train_examples else float('nan'),
                **{f'{k}': (v / train_examples if train_examples else float('nan'))
                for k, v in train_scores.items()}
            }
        }

        for k, v in train_scores.items():
            logger.info(f"{k} = {v / train_examples if train_examples else float('nan')}")

        if val_loader is not None:
            info['val'] = {
                'loss': val_loss / val_examples if val_examples else float('nan'),
                **{f'{k}': (v / val_examples if val_examples else float('nan'))
                for k, v in val_scores.items()}
            }

        actor.unfreeze()
        return info

    def forward(self, l):
        return (self.D(l), ) # return the logit value computed by the discriminator
    
    def loss(self, pred, label):
        domain_logits, = pred # the u_pred here has a shape of [64, 2], [batch_size, output_size]
        domain_ind, = label
        # Train discriminator to classify domain (1: source, 0: target)
        loss_fn = nn.BCEWithLogitsLoss()
        loss_d = loss_fn(domain_logits, domain_ind.float())  # domain_ind should be float [batch, 1]
        return loss_d, {'discriminator_loss': loss_d.item()}

class VisionAdversarialAdaptAC(BaseModel):
    """In this class, the agent and the discriminator are written as two different models"""

    feature_fields = ['camera', 'velocity']
    label_fields = ['action', "domain_indicator"]

    def __init__(self, pretrain_agent : VisionNaiveRandomization, pretrain_agent_params: dict, ad_agent_params: dict):
        """
        @ pretrain_encoder_path: the path to load the weight of the pretrained agent (the agent that will be adapted to the new environment)
        @ pretrain_agent_params: the parameters of the pretrained encoder
        @ ad_agent_params: the parameters of the discriminators, and some relevant modifications that should be made to the pretrain_agent params
        Model Input: states
        Model Output: actions
        domain_list: the list of source domains that the agent is trained with
        
        Loss: MSE
        """

        super().__init__()
        
        if pretrain_agent is None:
            return # null

        self.actor = VisionAdversarialActor(pretrain_agent=pretrain_agent, adv_factor=ad_agent_params['adv_factor']) # the adv_factor to the actor

        # initialize the discriminator
        self.discriminator = Discriminator(lr = ad_agent_params['lr_discriminator'], 
                                           weight_decay = ad_agent_params['weight_decay'],
                                           encoder_output_dim = ad_agent_params["encoder_output_dim"],
                                           dis_info_mode = ad_agent_params['dis_info_mode'])
        
        self.actor.set_discriminator(self.discriminator)

        self.actor.to(ptu.device)
        self.discriminator.to(ptu.device)
    
    def fit(self, train_dataset: sourceTargetBalanceBuffer, n_epochs, policy_val_dataset=None, disc_val_dataset = None, global_step=None):
        info = defaultdict(lambda: {})
        # first train the discriminator
        logger.info(f"///// Training the discriminator [{self.discriminator.model_name}] /////")

        # hard code the dis_epochs schedule for now
        dis_epochs = 8 if global_step == 0 else 3
        discriminator_info = self.discriminator.fit(n_epochs = dis_epochs, train_dataset = train_dataset, val_dataset=None, actor = self.actor)

        #train the actor
        logger.info(f"///// Training the actor ///// [{self.actor.model_name}]")
        actor_info = self.actor.fit(train_dataset=train_dataset, n_epochs = n_epochs, val_dataset = policy_val_dataset)

        info = (discriminator_info, actor_info)
        return info
    
    def get_action(self, *args):
        return self.actor.get_action(*args)
    
    def get_latent(self, img, to_numpy = True):
        return self.actor.get_latent(img, to_numpy)
    
    def parse_obs(self, *args):
        # forward the arguments to the actor 
        return self.actor.parse_obs(*args)

class VisionConditionAdversarialAdaptAC(VisionAdversarialAdaptAC):

    feature_fields = ['camera', 'velocity', 'state', 'curvature', 'gps']
    label_fields = ['action', "domain_indicator"]

    def __init__(self, pretrain_agent : VisionNaiveRandomization, pretrain_agent_params: dict, ad_agent_params: dict):
        """
        @ pretrain_encoder_path: the path to load the weight of the pretrained agent (the agent that will be adapted to the new environment)
        @ pretrain_agent_params: the parameters of the pretrained encoder
        @ ad_agent_params: the parameters of the discriminators, and some relevant modifications that should be made to the pretrain_agent params
        Model Input: states
        Model Output: actions

        Loss: MSE
        """
        super().__init__(pretrain_agent = None, pretrain_agent_params = None, ad_agent_params = None)

        if pretrain_agent == None:
            return # the null initialization
        
        self.actor = VisionConditionalAdversarialActor(pretrain_agent=pretrain_agent, adv_factor=ad_agent_params['adv_factor'])

        # initialize the discriminator
        # initialize the discriminator
        self.discriminator = CatDiscriminator(lr = ad_agent_params['lr_discriminator'], 
                                           weight_decay = ad_agent_params['weight_decay'],
                                           encoder_output_dim = ad_agent_params["encoder_output_dim"],
                                           dis_info_mode = ad_agent_params['dis_info_mode'])
        
        self.actor.set_discriminator(self.discriminator)

        self.actor.to(ptu.device)
        self.discriminator.to(ptu.device)

class VisionConditionalAdversarialActor(VisionAdversarialActor):
    feature_fields = ['camera', 'velocity', 'state', 'curvature', 'gps']
    label_fields = ['action', "domain_indicator"]

    def forward(self, img, vel, state = None, curvature = None, gps = None):

        # Normalize the image first. 
        img = img.permute(0, 3, 1, 2) / 255.

        l = self.resnet(img)

        # find the domain logits when discriminative information is passed in as side information
        if state is not None and curvature is not None and gps is not None:
            dis_info = self.discriminator.discriminative_info(state, curvature, gps)

            # logger.info(f"the disinfo = {dis_info}")
            
            domain_logits, = self.discriminator(l, dis_info)  # allow gradients to flow

            if self.check_latent_collapse(l):
                logger.info("WARNING: latent space collapse is detected!")
        else:
            logger.info("No discriminative info is passed in.")
            domain_logits = None

        v_encoded = self.velocity_encoder(vel)
        combined = torch.cat([l, v_encoded], dim=1)  # shape: [batch_size, latent_dim]

        output = self.decision(combined)
        return (output, domain_logits)

class CatDiscriminator(Discriminator):

    feature_fields = ['camera', 'state', 'curvature', 'gps']
    label_fields = ['domain_indicator']

    def __init__(self, lr, weight_decay, encoder_output_dim, dis_info_mode, null_init = False):
        super().__init__(lr=lr, weight_decay=weight_decay, encoder_output_dim = encoder_output_dim,
                         dis_info_mode= dis_info_mode, null_init = null_init)
        
        if self.dis_info_mode == "state_curvature":
            self.discriminative_info = self.h_state_curvature
        elif self.dis_info_mode == "only_curvature":
            self.discriminative_info = self.h_only_curvature
        elif self.dis_info_mode == "only_state":
            self.discriminative_info = self.h_only_state
        elif self.dis_info_mode == "only_x_tran":
            self.discriminative_info = self.h_only_x_tran
        elif self.dis_info_mode == "gps_xy":
            self.discriminative_info = self.h_global_xy
        elif self.dis_info_mode == 'gps_full':
            self.discriminative_info = self.h_gps_full
        else:
            raise ValueError("The input dis_info_mode is invalid")

    def input_buffer(self, features, actor : VisionAdversarialActor ):
        """preprocess the features before passing them to the model.
        note that in the dataloader, there are only RGB images, there is no such a field called latent vector"""

        img, state, curvature, gps = features
        dis_info = self.discriminative_info(state, curvature, gps)

        latent_vectors = actor.get_latent(img = img, to_numpy = False)
        latent_vectors = latent_vectors.detach() # cut from the rest of the computation graph

        return [latent_vectors, dis_info]
    
    def h_state_curvature(self, state, curvature, gps):
        """extract task relevant information from the state variable and the curvature variable,
        which is discriminative in all the domains"""
        task_info_state = state[:, -2:]        # shape: [batch_size, 2]
        task_info_curvature = curvature        # shape: [batch_size, 3]
        return torch.cat([task_info_state, task_info_curvature], dim=1)  # shape: [batch_size, 5]
    
    def h_global_xy(self, state, curvature, gps):
        """return the x, y global coordinates"""
        return gps[:, : 2]
    
    def h_gps_full(self, state, curvature, gps):
        """Return full GPS, with last column (e_psi) wrapped mod π."""
        gps[:, -1] = torch.remainder(gps[:, -1], math.pi)
        return gps
    
    def h_only_curvature(self, state, curvature, gps):
        """simply return curvature"""
        return curvature
    
    def h_only_state(self, state, curvature, gps):
        """only return e_psi and lateral deviation from the center line"""
        return state[:, -2:]
    
    def h_only_x_tran(self, state, curvature, gps):
        """only return x_tran, the lateral deviation as the discriminatie information"""
        return state[:, -2 : -1]

    def forward(self, l, dis_info):

        """The conditioning policy is simply concatenate the latent vectors with the discriminative information"""

        combined = torch.cat([l, dis_info], dim = 1)
        return (self.D(combined), ) # return the logit value computed by the discriminator

class DensityEstimator(BaseModel):
    
    def __init__(self, sample_distribution : str, pretrain_name : str):
        super().__init__()
        self.has_fit = False
        self.kde_src : KernelDensity = None
        self.kde_tgt : KernelDensity = None
        self.load_bandwidth_config()

        self.sample_distribution = sample_distribution
        self.pretrain_name = pretrain_name

    def load_bandwidth_config(self):
        # Get the directory of this script (i.e., adaptAC.py)
        current_dir = os.path.dirname(__file__)
        
        # Construct the absolute path to the YAML file
        yaml_path = os.path.join(current_dir, '..', 'config', 'best_bandwidth.yaml')

        # Normalize path (e.g., convert .. to full path)
        yaml_path = os.path.abspath(yaml_path)

        # Load YAML
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)

        self.best_bandwidth_config = config

    def save_bandwidth_config(self):
        """
        Save self.best_bandwidth_config to root_directory/config/best_bandwidth.yaml
        """
        # Get the directory where this script is located
        current_dir = os.path.dirname(__file__)

        # Construct the full path to the YAML file
        yaml_path = os.path.abspath(os.path.join(current_dir, '..', 'config', 'best_bandwidth.yaml'))

        # Write the config dictionary to the YAML file
        with open(yaml_path, 'w') as f:
            yaml.dump(self.best_bandwidth_config, f, default_flow_style=False, sort_keys=False)

        print(f"Saved best_bandwidth_config to {yaml_path}")


    def dis_info_collate(self, dataset: EfficientReplayBuffer):
        state = dataset.retrieve_entire_field("state")[:, -2:]
        curvature = dataset.retrieve_entire_field("curvature")
        dis_info = np.concatenate([state, curvature], axis = 1)
        return dis_info
    
    def best_bandwidth(self, data, start: float, end: float, domain: str):
        if domain == 'target':
            key = self.sample_distribution + '_' + str(data.shape[0])
        elif domain == 'source':
            key = self.pretrain_name
        else:
            raise ValueError(f"the domain variable must be either target or source.")
        
        """first try to locate the bandwidth inside self.best_bandwidth_config, if it is found there, directly return it;
        otherwise, gridsearch it and rewrite it to the config file"""

        if key in self.best_bandwidth_config[domain]:
            logger.info(f"Found the best bandiwidth configure for {domain} buffer with key {key}")
            return self.best_bandwidth_config[domain][key]

        # Choose a range of candidate bandwidths
        bandwidths = np.linspace(start, end, 10)

        grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                            {'bandwidth': bandwidths},
                            cv=3)  # 3-fold cross-validation

        grid.fit(data)  # X = your dataset (source or target)
        best_h = grid.best_params_['bandwidth']
        
        self.best_bandwidth_config[domain][key] = float(best_h)
        self.save_bandwidth_config()

        return best_h
    
    def sigmoid_squash(self, x, bias, var, min_val=0.1, max_val=10):
        """soft clipping strategy: used to regularize the range of density ratios while keeping the distribution 
        structure of the density ratios"""
        scaled = (x - bias) / (var + 1e-8)
        return min_val + (max_val - min_val) * (1 / (1 + np.exp(-scaled)))

    def fit(self, dataset: sourceTargetBalanceBuffer, display_ratio = False):
        if self.has_fit:
            logger.warning("The Density Ratio Estimator should only be fit for one time!")
        
        logger.info("Fitting the density ratio estimator...")
        dis_info_source = self.dis_info_collate(dataset.source_buffer)
        dis_info_target = self.dis_info_collate(dataset.target_buffer)

        source_bandwidth = self.best_bandwidth(dis_info_source, start = 0.01, end = 0.2, domain = 'source')
        target_bandwidth = self.best_bandwidth(dis_info_target, start = 0.01, end = 0.2, domain = 'target')

        # source_bandwidth = 0.04 # the empirical value for the dataset of pretrain_bright3
        # target_bandwidth = 0.055 # the empirical value for first4mDistribution with a size of 512

        logger.info(f"source domain bandwidth = {source_bandwidth}; target domain bandwidth = {target_bandwidth}")

        self.kde_src = KernelDensity(kernel='gaussian', bandwidth=source_bandwidth).fit(dis_info_source)
        self.kde_tgt = KernelDensity(kernel='gaussian', bandwidth=target_bandwidth).fit(dis_info_target)
        logger.info(f"The ratio estimator is now fit.")
        self.has_fit = True

        self.st_ratios_target_mean = np.mean(self.sourceTargetRatios(dis_info = dis_info_target, to_tensor = False, clipped=False, normalize = False))

        if display_ratio:
            # Test the distribution of probability densities calculated by the density_estimator
            tgt_display_ratios = self.sourceTargetRatios(dis_info = dis_info_target, to_tensor = False, clipped = False)
            plt.figure(figsize=(8, 5))
            plt.hist(tgt_display_ratios, bins=50, edgecolor='black')
            plt.title("Histogram of Density Ratios (Source / Target)")
            plt.xlabel("Density Ratio")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.show()
    
    def densities(self, dis_info):

        if isinstance(dis_info, torch.Tensor):
            dis_info = ptu.clone_to_numpy()

        source_est = self.kde_src.score_samples(dis_info)
        target_est = self.kde_tgt.score_samples(dis_info)

        return source_est, target_est
    
    def sourceTargetRatios(self, dis_info, epsilon: float = 1e-8, normalize: bool = True, to_tensor : bool = True, clipped : bool = True):
        """
        Compute density ratios w(x) = p_source(x) / (p_target(x) + epsilon)
        
        Args:
            dis_info (np.ndarray or torch.Tensor): Input features of shape [B, D]
            epsilon (float): Small value to avoid division by zero (log-domain)
            normalize (bool): Whether to normalize the output weights to have mean 1

        Returns:
            np.ndarray: Density ratios of shape [B]
        """

        if self.kde_src is None or self.kde_tgt is None:
            return torch.tensor([0.]) # Null return if the kde estimator has not been initialized
        
        if isinstance(dis_info, torch.Tensor):
            dis_info = ptu.clone_to_numpy(dis_info)

        log_p_src, log_p_tgt = self.densities(dis_info)
        
        # Stabilize denominator by clipping log_p_tgt
        log_p_tgt_clipped = np.maximum(log_p_tgt, np.log(epsilon))

        # Compute log ratio, then exponentiate
        log_ratio = log_p_src - log_p_tgt_clipped

        ratio = np.exp(log_ratio)

        if normalize:
            weight = np.log1p(ratio) / self.st_ratios_target_mean # do log(1 + r) to smooth values smaller than 1
        else:
            weight = np.log1p(ratio)

        if clipped:
            weight = np.clip(weight, 0.75, 10) # clip the weight values to prevent gradient explosion

        if to_tensor:
            weight = ptu.from_numpy(weight)

        return weight