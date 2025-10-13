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

class VisionSafeActor(BaseModel):
    feature_fields = ['camera', 'velocity']
    label_fields = ['action', 'state']

    def __init__(self, ob_dim, ac_dim, size, n_layers, lr=1e-3, weight_decay=1e-5, critic=None, dynamics=None, lam=1.):
        """
        Model Input: states
        Model Output: actions

        Loss: MSE + NLL
        """
        super().__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Sequential()

        self.mlp = ptu.build_mlp(input_size=512 + ob_dim, output_size=ac_dim, size=size, n_layers=n_layers,
                                 activation='relu', )
        self.critic = critic  # The critic outputs logits (without sigmoid).
        self.dynamics = dynamics

        self.optimizer = Adam(itertools.chain(self.resnet.parameters(), self.mlp.parameters()), lr=lr,
                              weight_decay=weight_decay)
        self.scheduler = ExponentialLR(optimizer=self.optimizer, gamma=1)  # 0.1 ** (1 / 100))
        self.loss_func = nn.MSELoss()
        self.log_sigmoid = nn.LogSigmoid()
        # self.img_mean = ptu.from_numpy(np.array([0.485, 0.456, 0.406])).view(1, 3, 1, 1).float()
        # self.img_std = ptu.from_numpy(np.array([0.229, 0.224, 0.225])).view(1, 3, 1, 1).float()
        self.lam = lam

    def step_schedule(self):
        self.lam = min(10., self.lam * 1.007)  # Approx 2 ** (1 / 100)

    def loss(self, pred, label, k=1, temperature=1):
        u_pred, = pred # the u_pred here has a shape of [64, 2], [batch_size, output_size]
        u, states = label
        mse_loss = self.loss_func(u_pred, u)
        if not self.critic.initialized:
            return mse_loss, {'mse_loss': mse_loss.item()}
        x_next_pred, = self.dynamics(states, u_pred)
        logits, = self.critic(x_next_pred)
        nll_loss = -self.lam * self.log_sigmoid(logits / temperature).sum() / logits.size(0)
        #Note: the loss here is the average loss over the entire batch
        return mse_loss + nll_loss, {'mse_loss': mse_loss.item(), 'nll_loss': nll_loss.item()}  # MSE + NLL.

    def forward(self, img, vel):
        # Normalize the image first. 
        img = img.permute(0, 3, 1, 2) / 255.
        # img = (img / 255. - self.img_mean) / self.img_std
        # logger.debug(f"{img.size()}")
        l = self.resnet(img)

        out = self.mlp(torch.cat([l, vel], dim=1))

        return (out,)

class VisionContrastiveActor(VisionSafeActor):
    feature_fields = ['camera', 'velocity', "positive", "negatives"]
    carla_fields = ["camera", "velocity"]
    label_fields = ['action', 'state']
    alpha = 0.3 # the paramter used to scale the contrastive_loss

    def parse_carla_obs(self, obs, info):
        """This function is the interface between the carla environment and the actor
        Note, in the carla observation, there is no field of positive and negatives"""
        
        try:
            out = [obs[field] for field in self.carla_fields] + [info[field] for field in
                                                                    self.additional_input_fields]
            return out
        except KeyError as e:
            logger.error(f"e")
            logger.error(f"Available fields are: {list(obs.keys()) + list(info.keys())}")

    def get_action(self, *args):
        self.eval()
        outputs = [x for x in self(*[ptu.from_numpy(arg.copy()[None]) for arg in args])]
        for i, output in enumerate(outputs):
            outputs[i] = ptu.to_numpy(output)
        u_pred = outputs[0][0]
        return u_pred
    
    def loss(self, pred, label, k=1, temperature=1):
 
        u_pred, l, pos_l, neg_ls = pred  # l, pos_l: [B, D], neg_ls: [B, L, D]
        u, states = label

        mse_loss = self.loss_func(u_pred, u)

        # === Add InfoNCE loss ===
        # Normalize for cosine similarity (optional, but common)
        l_norm = F.normalize(l, dim=1)                   # [B, D]
        pos_l_norm = F.normalize(pos_l, dim=1)           # [B, D]
        neg_ls_norm = F.normalize(neg_ls, dim=2)         # [B, L, D]

        # l_norm = l
        # pos_l_norm = pos_l
        # neg_ls_norm = neg_ls

        # Positive logits: [B, 1]
        pos_logits = torch.sum(l_norm * pos_l_norm, dim=1, keepdim=True)  # [B, 1]

        # Negative logits: [B, L]
        neg_logits = torch.bmm(neg_ls_norm, l_norm.unsqueeze(2)).squeeze(2)  # [B, L]

        # Combine: [B, 1 + L]
        logits = torch.cat([pos_logits, neg_logits], dim=1) / temperature

        # Targets: index 0 is the positive
        targets = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        contrastive_loss = self.alpha * F.cross_entropy(logits, targets)  # InfoNCE 

        # === Existing NLL loss ===
        if not self.critic.initialized:
            return mse_loss + contrastive_loss, {'mse_loss': mse_loss.item(), 'contrastive_loss': contrastive_loss.item()}

        x_next_pred, = self.dynamics(states, u_pred)
        logits_nll, = self.critic(x_next_pred)
        nll_loss = -self.lam * self.log_sigmoid(logits_nll / temperature).sum() / logits_nll.size(0)

        # Total loss
        total_loss = mse_loss + nll_loss + contrastive_loss  # Add alpha as weight if needed

        return total_loss, {
        'mse_loss': mse_loss.item(),
        'nll_loss': nll_loss.item(),
        'contrastive_loss': contrastive_loss.item()
        }

    def forward(self, img, vel, pos_img = None, neg_imgs = None):
        # Normalize the image first. 
        # Normalize and permute [B, H, W, 3] -> [B, 3, H, W]
        img = img.permute(0, 3, 1, 2) / 255.0
        l = self.resnet(img) # [B, D]

        if pos_img is not None and neg_imgs is not None:
            pos_img = pos_img.permute(0, 3, 1, 2) / 255.0
            # Reshape neg_imgs: [B, L, H, W, 3] -> [B*L, 3, H, W]
            B, L, H, W, C = neg_imgs.shape
            neg_imgs = neg_imgs.view(B * L, H, W, C).permute(0, 3, 1, 2) / 255.0

            # Compute latent features
            pos_l = self.resnet(pos_img)         # [B, D]
            neg_ls_flat = self.resnet(neg_imgs)  # [B*L, D]

            # Reshape back to [B, L, D]
            D = neg_ls_flat.shape[1]
            neg_ls = neg_ls_flat.view(B, L, D)   # [B, L, D]

        else:
            pos_l = torch.zeros(1)
            neg_ls = torch.zeros(1)
        
        # Main task head
        out = self.mlp(torch.cat([l, vel], dim=1))  # [B, output_dim]

        return (out, l, pos_l, neg_ls)


class VisionContrastiveAC(SafeAC, nn.Module):
    def __init__(self, st_dim, ob_dim, ac_dim, n_layers, size, lr=1e-3, weight_decay=1e-5,
                 dyn_size=3, dyn_layers=64, critic_size=64, critic_layers=4,
                 lam=1.,):
        torch.cuda.empty_cache()
        super().__init__(st_dim=st_dim, ac_dim=ac_dim, n_layers=n_layers, size=size)

        self.dynamics = Dynamics(st_dim=st_dim, ac_dim=ac_dim, size=dyn_size, n_layers=dyn_layers,
                                 lr=lr, weight_decay=weight_decay)
        self.critic = SafeCritic(st_dim=st_dim, ac_dim=ac_dim, size=critic_size, n_layers=critic_layers,
                                 lr=lr, weight_decay=weight_decay)
        self.actor = VisionContrastiveActor(ob_dim=ob_dim, ac_dim=ac_dim, size=size, n_layers=n_layers,
                                     lr=lr, weight_decay=weight_decay,
                                     lam=lam, critic=self.critic, dynamics=self.dynamics)


class VisionSafeAC(SafeAC, nn.Module):
    def __init__(self, st_dim, ob_dim, ac_dim, n_layers, size, lr=1e-3, weight_decay=1e-5,
                 dyn_size=3, dyn_layers=64, critic_size=64, critic_layers=4,
                 lam=1.,):
         
        super().__init__(st_dim=st_dim, ac_dim=ac_dim, n_layers=n_layers, size=size)

        self.dynamics = Dynamics(st_dim=st_dim, ac_dim=ac_dim, size=dyn_size, n_layers=dyn_layers,
                                 lr=lr, weight_decay=weight_decay)
        self.critic = SafeCritic(st_dim=st_dim, ac_dim=ac_dim, size=critic_size, n_layers=critic_layers,
                                 lr=lr, weight_decay=weight_decay)
        self.actor = VisionSafeActor(ob_dim=ob_dim, ac_dim=ac_dim, size=size, n_layers=n_layers,
                                     lr=lr, weight_decay=weight_decay,
                                     lam=lam, critic=self.critic, dynamics=self.dynamics)

    def fit(self, train_dataset: EfficientReplayBufferPN, n_epochs, val_dataset=None, global_step=None):
        info = defaultdict(lambda: {})
        if global_step is None or global_step % 5 == 0:
            dynamics_info = self.dynamics.fit(train_dataset=train_dataset, n_epochs=n_epochs * 10)
        else:
            dynamics_info = {}

        if train_dataset.D_neg.initialized and len(train_dataset.D_neg) > 16 and (global_step is None or global_step % 10 == 0):
            self.critic.initialized = True
            critic_info = self.critic.fit(train_dataset=train_dataset, n_epochs=n_epochs * 10)
        else:
            critic_info = {}
        actor_info = self.actor.fit(train_dataset=train_dataset.D_pos, n_epochs=n_epochs)
        for d in (dynamics_info, actor_info, critic_info):
            for k1, v1 in d.items():
                for k2, v2 in v1.items():
                    info[k1][k2] = v2
        return info

class VisionNaiveRandomization(BaseModel):
    feature_fields = ['camera', 'velocity']
    label_fields = ['action', 'state']

    def __init__(self, ob_dim, ac_dim, size, n_layers, lr=1e-3, weight_decay=1e-5, lam=1., l_dim = 10):
        """
        Model Input: states
        Model Output: actions

        Loss: MSE
        """
        super().__init__()
        self.initialize_encoders(ob_dim)

        self.decision = nn.Linear(512 + 16, ac_dim)
        logger.info(f"The learning rate = {lr}")

        self.optimizer = Adam(itertools.chain(self.resnet.parameters(), self.decision.parameters(), self.velocity_encoder.parameters()), lr=lr,
                              weight_decay=weight_decay)
        self.scheduler = ExponentialLR(optimizer=self.optimizer, gamma=1)  # 0.1 ** (1 / 100))
        self.loss_func = nn.MSELoss()
        self.log_sigmoid = nn.LogSigmoid()
        self.lam = lam
    
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


class VisionNaiveRandomization_Visualization(BaseModel):
    """This class is used for visualizing the latent space distribution"""
    feature_fields = ['camera', 'velocity']
    label_fields = ['action', 'state']

    def __init__(self, ob_dim, ac_dim, size, n_layers, lr=1e-3, weight_decay=1e-5, lam=1., l_dim = 10):
        """
        Model Input: states
        Model Output: actions

        Loss: MSE
        """
        super().__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Sequential(
             nn.Linear(512, 256),
             nn.ReLU(),
             nn.Linear(256, 2) # this bottleneck design is only for visualization.
        )

        self.mlp = nn.Linear(2 + ob_dim, ac_dim)

        # self.resnet.fc = nn.Sequential()
        # size = 128
        # self.mlp = ptu.build_mlp(input_size=512 + ob_dim, output_size=ac_dim, size=size, n_layers=n_layers,
        #                          activation='relu', )

        self.optimizer = Adam(itertools.chain(self.resnet.parameters(), self.mlp.parameters()), lr=lr,
                              weight_decay=weight_decay)
        self.scheduler = ExponentialLR(optimizer=self.optimizer, gamma=1)  # 0.1 ** (1 / 100))
        self.loss_func = nn.MSELoss()
        self.log_sigmoid = nn.LogSigmoid()
        # self.img_mean = ptu.from_numpy(np.array([0.485, 0.456, 0.406])).view(1, 3, 1, 1).float()
        # self.img_std = ptu.from_numpy(np.array([0.229, 0.224, 0.225])).view(1, 3, 1, 1).float()
        self.lam = lam

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
        out = self.mlp(torch.cat([l, vel], dim=1))
        return (out,)
    
    def get_latent(self, img):
        self.eval()
        img = img.permute(0, 3, 1, 2) / 255.
        l = self.resnet(img)

        return ptu.to_numpy(l)

class VisionNaiveMultihead(BaseModel):

    """the most naive architecture of multihead model, we assume that we have perfect knowledge about the
    domain of the input image, the domain index is not obtained via any domain classifier,
    Instead, it will be taken in directly as a feature for the model to make decisions"""

    feature_fields = ['camera', 'velocity', "domain_v"]
    label_fields = ['action', 'state']

    def __init__(self, pretrained_agent, ob_dim, ac_dim, size, n_layers, lr=1e-3, weight_decay=1e-5, lam=1., l_dim = 10, heads_num = 3):
        """
        Model Input: states
        Model Output: actions

        Loss: MSE
        """
        super().__init__()
        self.pretrain_agent = pretrained_agent
        
        # use the visual encoders of the trained agent directly
        self.resnet = self.pretrain_agent.resnet
        self.velocity_encoder = self.pretrain_agent.velocity_encoder
        initial_decision_head = self.pretrain_agent.decision
        
        resnet_output_shape = self.get_output_shape(self.resnet, (1, 3, 224, 224))
        VE_output_shape = self.get_output_shape(self.velocity_encoder, (1, ob_dim))
        logger.info(f"/////// The loaded pretrained encoders has the following output shapes: visual_encoder = {resnet_output_shape}; velocity_encoder = {VE_output_shape}")

        # Multiple decision heads
        self.heads_num = heads_num

        self.decision_heads = nn.ModuleList([
            copy.deepcopy(initial_decision_head) for _ in range(heads_num)
        ])

        del self.pretrain_agent # delete the pretrained agent to release memory

        # Optimizer and scheduler
        self.optimizer = Adam(itertools.chain(
            self.resnet.parameters(),
            self.velocity_encoder.parameters(),
            *[head.parameters() for head in self.decision_heads]
        ), lr=lr, weight_decay=weight_decay)
        
        self.scheduler = ExponentialLR(optimizer=self.optimizer, gamma=1)  # 0.1 ** (1 / 100))
        self.loss_func = nn.MSELoss()
        self.log_sigmoid = nn.LogSigmoid()
        self.lam = lam
        self.freeze_encoders() # automatically freeze the encoders
    
    def get_output_shape(self, encoder, input_shape):
        """this method is used to get the output shapes of the encoder"""
        dummy_input = torch.zeros(input_shape)

        with torch.no_grad():
            dummy_output = encoder(dummy_input)

        return dummy_output.shape[1]


    def step_schedule(self):
        self.lam = min(10., self.lam * 1.007)  # Approx 2 ** (1 / 100)
    
    def loss(self, pred, label, k=1, temperature=1):
        u_pred, = pred # the u_pred here has a shape of [64, 2], [batch_size, output_size]
        u, states = label
        mse_loss = self.loss_func(u_pred, u)
        return mse_loss, {'mse_loss': mse_loss.item()}
    
    def forward(self, img, vel, domain_v):
        # Normalize the image first. 
        img = img.permute(0, 3, 1, 2) / 255.
        # img = (img / 255. - self.img_mean) / self.img_std
        # logger.debug(f"{img.size()}")
        l = self.resnet(img)

        if self.check_latent_collapse(l):
            logger.info("WARNING: latent space collapse is detected!")

        v_encoded = self.velocity_encoder(vel)
        combined = torch.cat([l, v_encoded], dim=1)  # shape: [batch_size, latent_dim]

        # Pass through all heads and stack outputs: shape [batch_size, heads_num, ac_dim]
        all_outputs = torch.stack([
        head(combined) for head in self.decision_heads
        ], dim=1)

        # logger.info(f"the dimension of all_outputs is {all_outputs.shape}")

        # Perform weighted sum across heads using domain probabilities
        # domain_index: [batch_size, heads_num]
        # all_outputs:  [batch_size, heads_num, ac_dim]
        # → weighted sum over heads → [batch_size, ac_dim]
        domain_probs = domain_v.unsqueeze(-1)  # [batch_size, heads_num, 1]

        # logger.info(f"the dimension of the domain_probs is {domain_probs.shape}")

        weighted_output = (domain_probs * all_outputs).sum(dim=1)

        # logger.info(f"the dimension of the weighted output is {weighted_output.shape}")
        return (weighted_output,)
    
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

        # Reinitialize optimizer to only update decision heads
        self.optimizer = Adam(
            itertools.chain(*[head.parameters() for head in self.decision_heads]),  # all heads in the ModuleList
            lr=self.optimizer.param_groups[0]['lr'],
            weight_decay=self.optimizer.param_groups[0]['weight_decay']
    )

class VisionCompleteMultiHead(BaseModel):

    """built on the basis of naive multihead structure, instead of assuming that we have perfect knowledge about the
    domain index, the probaility vector will be calculated by a pretrained classifier
    For this architecture, the classifier and the decision layers need to be trained separately."""

    feature_fields = ['camera', 'velocity']
    label_fields = ['action', 'state', "domain_v"]
    modes = ['only_classifier', 'full_prediction'] # choose one mode from these two modes

    def __init__(self, pretrained_agent, ob_dim, ac_dim, size, n_layers, lr=1e-3, weight_decay=1e-5, lam=1., l_dim = 10, heads_num = 3, mode = 'only_classifier'):
        """
        Model Input: states
        Model Output: actions

        Loss: MSE
        """
        super().__init__()
        self.pretrain_agent = pretrained_agent
        
        # use the visual encoders of the trained agent directly
        self.resnet = self.pretrain_agent.resnet
        self.velocity_encoder = self.pretrain_agent.velocity_encoder
        initial_decision_head = self.pretrain_agent.decision
        
        resnet_output_shape = self.get_output_shape(self.resnet, (1, 3, 224, 224))
        VE_output_shape = self.get_output_shape(self.velocity_encoder, (1, ob_dim))
        logger.info(f"/////// The loaded pretrained encoders has the following output shapes: visual_encoder = {resnet_output_shape}; velocity_encoder = {VE_output_shape}")

        # Multiple decision heads
        self.heads_num = heads_num

        self.decision_heads = nn.ModuleList([
            copy.deepcopy(initial_decision_head) for _ in range(heads_num)
        ])

        del self.pretrain_agent # delete the pretrained agent to release memory

        self.classifier = resnet18(weights=ResNet18_Weights.DEFAULT) # This architecture uses the classifier as a seperate module
        self.classifier.fc = nn.Linear(512, self.heads_num)

        # freeze the visual encoder
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Freeze velocity encoder
        for param in self.velocity_encoder.parameters():
            param.requires_grad = False

        # Reinitialize optimizer to only update decision heads and the classifier
        self.optimizer = Adam(
            itertools.chain(self.classifier.parameters(), *[head.parameters() for head in self.decision_heads]),  # all heads in the ModuleList
            lr=lr, weight_decay=weight_decay
    )
        
        self.scheduler = ExponentialLR(optimizer=self.optimizer, gamma=1)  # 0.1 ** (1 / 100))
        self.loss_func = nn.MSELoss()

        if mode not in self.modes:
            raise TypeError("The input mode is not in compatible mode list, choose one from ['only_classifier', 'full_prediction']")
        
        self.mode = mode
    
    def get_output_shape(self, encoder, input_shape):
        """this method is used to get the output shapes of the encoder"""
        dummy_input = torch.zeros(input_shape)

        with torch.no_grad():
            dummy_output = encoder(dummy_input)

        return dummy_output.shape[1]

    def step_schedule(self):
        self.lam = min(10., self.lam * 1.007)  # Approx 2 ** (1 / 100)
    
    def to_decision_training(self):
        logger.info("Switching to train the decision layers !!!! the classifier is no longer trainable")
        self.mode = self.modes[1]
        # Freeze the classifier
        for param in self.classifier.parameters():
            param.requires_grad = False

        # Reinitialize optimizer to only update decision heads
        self.optimizer = Adam(
            itertools.chain(*[head.parameters() for head in self.decision_heads]),  # all heads in the ModuleList
            lr=self.optimizer.param_groups[0]['lr'],
            weight_decay=self.optimizer.param_groups[0]['weight_decay']
    )
    
    def loss(self, pred, label):
        u_pred, domain_probs_pred = pred # the u_pred here has a shape of [64, 2], [batch_size, output_size]
        u, states, domain_probs = label

        if self.mode == self.modes[0]:
            mse_loss = self.loss_func(domain_probs_pred, domain_probs)

        elif self.mode == self.modes[1]:
            mse_loss = self.loss_func(u_pred, u)

        return mse_loss, {'mse_loss': mse_loss.item()}
    
    def forward(self, img, vel):

        # Normalize the image first. 
        img = img.permute(0, 3, 1, 2) / 255.

        # domain_logits: [batch_size, heads_num]
        domain_logits = self.classifier(img)  # feed the raw image to domain classifier
        domain_probs = F.softmax(domain_logits, dim=1)  # [batch_size, heads_num]

        l = self.resnet(img)

        if self.check_latent_collapse(l):
            logger.info("WARNING: latent space collapse is detected!")

        v_encoded = self.velocity_encoder(vel)
        combined = torch.cat([l, v_encoded], dim=1)  # shape: [batch_size, latent_dim]

        # Pass through all heads and stack outputs: shape [batch_size, heads_num, ac_dim]
        all_outputs = torch.stack([
            head(combined) for head in self.decision_heads
            ], dim=1)

        # Perform weighted sum across heads using domain probabilities
        # domain_probs: [batch_size, heads_num] -> unsqueeze to the shape of [batch_size, heads_num, 1]
        # all_outputs:  [batch_size, heads_num, ac_dim]
        # → weighted sum over heads → [batch_size, ac_dim]
        domain_v = domain_probs.unsqueeze(-1)

        weighted_output = (domain_v * all_outputs).sum(dim=1)
        return (weighted_output, domain_probs)
    
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
        
    def get_action(self, *args):
        self.eval()
        outputs = [x for x in self(*[ptu.from_numpy(arg.copy()[None]) for arg in args])]
        for i, output in enumerate(outputs):
            outputs[i] = ptu.to_numpy(output)
        u_pred, domain_probs_pred = outputs[0][0], outputs[1][0]
        return u_pred, domain_probs_pred
    
    def get_latent(self, img):
        self.eval()
        img = img.permute(0, 3, 1, 2) / 255.
        l = self.resnet(img)

        return ptu.to_numpy(l)

class VisionAdversarialAdaptActor_Modes(BaseModel):

    """built on the basis of naive multihead structure, instead of assuming that we have perfect knowledge about the
    domain index, the probaility vector will be calculated by a pretrained classifier
    For this architecture, the classifier and the decision layers need to be trained separately.
    This adversarial adaptor is written by putting the dscriminator and the agent at the same class"""

    feature_fields = ['camera', 'velocity']
    label_fields = ['action', 'state', "domain_indicator"]
    modes = ['actor', 'discriminator'] # choose one mode from these two modes

    def __init__(self, pretrained_agent, ob_dim, ac_dim, size, n_layers, lr_d=1e-3, lr_a = 5e-5, weight_decay=1e-5, lam=1., l_dim = 10, mode = 'discriminator'):
        """
        Model Input: states
        Model Output: actions

        Loss: MSE
        """
        super().__init__()
        self.pretrain_agent = pretrained_agent
        
        # use the visual encoders of the trained agent directly
        self.resnet = self.pretrain_agent.resnet
        self.velocity_encoder = self.pretrain_agent.velocity_encoder
        initial_decision_head = self.pretrain_agent.decision
        
        resnet_output_shape = self.get_output_shape(self.resnet, (1, 3, 224, 224))
        VE_output_shape = self.get_output_shape(self.velocity_encoder, (1, ob_dim))
        logger.info(f"/////// The loaded pretrained encoders has the following output shapes: visual_encoder = {resnet_output_shape}; velocity_encoder = {VE_output_shape}")

        self.decision = initial_decision_head

        #initialize the discriminator
        # self.discriminator = nn.Linear(resnet_output_shape, 1)
        self.discriminator = nn.Sequential( # try deeper architecture for the discriminator
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
        )

        # Freeze velocity encoder
        for param in self.velocity_encoder.parameters():
            param.requires_grad = False

        self.actor_optimizer = Adam(itertools.chain(self.resnet.parameters(), self.decision.parameters()), lr=lr_a, # lr_a is the learning rate for the actor and lr_d is the learning rate for the discriminator
                              weight_decay=weight_decay)
        
        self.actor_scheduler = ExponentialLR(optimizer=self.actor_optimizer, gamma=1)  # 0.1 ** (1 / 100))

        self.discriminator_optimizer = Adam(self.discriminator.parameters(), lr = lr_d, weight_decay = weight_decay)
        self.discriminator_scheduler = ExponentialLR(optimizer=self.discriminator_optimizer, gamma=1)
        
        self.optimizers = [self.actor_optimizer, self.discriminator_optimizer]
        self.schedulers = [self.actor_scheduler, self.discriminator_scheduler]

        self.loss_func = nn.MSELoss()
        self.log_sigmoid = nn.LogSigmoid()
        self.lam = lam
        
        self.switch_mode(mode)
    
    def get_output_shape(self, encoder, input_shape):
        """this method is used to get the output shapes of the encoder"""
        dummy_input = torch.zeros(input_shape)

        with torch.no_grad():
            dummy_output = encoder(dummy_input)

        return dummy_output.shape[1]
    
    def step_schedule(self):
        self.lam = min(10., self.lam * 1.007)  # Approx 2 ** (1 / 100)

    
    def lazy_switch(self):
        switch_index = 1 - self.modes.index(self.mode)
        self.switch_mode(new_mode = self.modes[switch_index])
    
    def switch_mode(self, new_mode):
        """switch self.mode = new_mode, and adjust the optimizor and the scheduler accordingly"""
        if new_mode not in self.modes:
            raise TypeError(f"The input mode is not in compatible mode list, choose one from {self.modes}")
        self.mode = new_mode
        logger.info(f"Switching to the mode: {self.mode}")
        mode_index = self.modes.index(new_mode)

        self.optimizer = self.optimizers[mode_index]
        self.scheduler = self.schedulers[mode_index]

        if new_mode == 'actor':
            # Enable gradients for resnet and decision layers
            for param in self.resnet.parameters():
                param.requires_grad = True
            for param in self.decision.parameters():
                param.requires_grad = True
            # Disable gradients for discriminator
            for param in self.discriminator.parameters():
                param.requires_grad = False

        elif new_mode == 'discriminator':
            # Disable gradients for resnet and decision layers
            for param in self.resnet.parameters():
                param.requires_grad = False
            for param in self.decision.parameters():
                param.requires_grad = False
            # Enable gradients for discriminator
            for param in self.discriminator.parameters():
                param.requires_grad = True
    
    def loss(self, pred, label):
        u_pred, domain_logits = pred # the u_pred here has a shape of [64, 2], [batch_size, output_size]
        u, states, domain_ind = label

        if self.mode == 'discriminator':
            # Train discriminator to classify domain (1: source, 0: target)
            loss_fn = nn.BCEWithLogitsLoss()
            loss_d = loss_fn(domain_logits, domain_ind.float())  # domain_ind should be float [batch, 1]
            return loss_d, {'discriminator_loss': loss_d.item()}

        elif self.mode == 'actor':
            # Policy loss
            policy_loss = self.loss_func(u_pred, u)

            # Adversarial loss: encourage encoder to fool discriminator
            # Flip domain labels: try to make domain_ind_pred look like the opposite (e.g., 0.5 or source)
            target_labels = 1. - domain_ind.float()  # invert labels
            adv_loss_fn = nn.BCEWithLogitsLoss()
            adv_loss = adv_loss_fn(domain_logits, target_labels)

            total_loss = policy_loss + 0.1 * adv_loss
            return total_loss, {
                'policy_loss': policy_loss.item(),
                'adversarial_loss': adv_loss.item(),
                'total_loss': total_loss.item()
            }
    
    def forward(self, img, vel):

        # Normalize the image first. 
        img = img.permute(0, 3, 1, 2) / 255.

        l = self.resnet(img)

        # get the domain indicator produced by the discriminator
        # Conditionally detach based on mode
        if self.mode == 'discriminator':
            domain_logits = self.discriminator(l.detach())
        else:
            domain_logits = self.discriminator(l)  # allow gradients to flow

        if self.check_latent_collapse(l):
            logger.info("WARNING: latent space collapse is detected!")

        v_encoded = self.velocity_encoder(vel)
        combined = torch.cat([l, v_encoded], dim=1)  # shape: [batch_size, latent_dim]

        output = self.decision(combined)
        return (output, domain_logits)
    
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
        
    def get_action(self, *args):
        self.eval()
        outputs = [x for x in self(*[ptu.from_numpy(arg.copy()[None]) for arg in args])]
        for i, output in enumerate(outputs):
            outputs[i] = ptu.to_numpy(output)
        u_pred, domain_label = outputs[0][0], outputs[1][0]
        return u_pred, domain_label
    
    def get_latent(self, img):
        self.eval()
        img = img.permute(0, 3, 1, 2) / 255.
        l = self.resnet(img)

        return ptu.to_numpy(l)

class VisionAdversarialActor(BaseModel):
    """Unlike VisionAdversarialAdaptActor_modes, this adversarial actor is not implemented based on the mode mechanism"""
    feature_fields = ['camera', 'velocity']
    label_fields = ['action', "domain_indicator"]

    def __init__(self, pretrain_agent : VisionNaiveRandomization, adv_factor = 1.0):
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

        # freeze the velocity encoder
        # for p in self.velocity_encoder.parameters():
        #     p.requires_grad = False
    
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
        if dis_info_mode == "only_curvature":
            dis_info_dim = 3
        elif dis_info_mode == "state_curvature":
            dis_info_dim = 5
        elif dis_info_mode == "only_state" or dis_info_mode == "gps":
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

        Loss: MSE
        """

        super().__init__()
        
        if pretrain_agent is None:
            return # null

        self.actor = VisionAdversarialActor(pretrain_agent=pretrain_agent)

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
        if global_step == 0:
            dis_epochs = 12
        
        elif global_step == 44: # the time at which the adversarial game is stuck with Nush Equilibrium, interrupt with the game
            dis_epochs = 9
        else:
            dis_epochs = 3

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
        
        self.actor = VisionConditionalAdversarialActor(pretrain_agent=pretrain_agent)

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

    def forward(self, img, vel, state, curvature, gps):

        # Normalize the image first. 
        img = img.permute(0, 3, 1, 2) / 255.

        l = self.resnet(img)
        dis_info = self.discriminator.discriminative_info(state, curvature, gps)

        # logger.info(f"the disinfo = {dis_info}")
        
        domain_logits, = self.discriminator(l, dis_info)  # allow gradients to flow

        if self.check_latent_collapse(l):
            logger.info("WARNING: latent space collapse is detected!")

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
        elif self.dis_info_mode == "gps":
            self.discriminative_info = self.h_global_xy
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

class ProjDiscriminator(CatDiscriminator):
    """More advanced version of conditioning policy, 
    exploiting the possibility of using projection network to realize the conditioning effect"""

    feature_fields = ['camera', 'state', 'curvature']
    label_fields = ['domain_indicator']

    def __init__(self, lr, weight_decay, encoder_output_dim, dis_info_dim):

        super().__init__(lr = lr, weight_decay = weight_decay, encoder_output_dim = encoder_output_dim,
                          dis_info_dim = dis_info_dim, null_init=True) # null initialization of the super class
        proj_dim = 128 # the dimension of the prjected common space for latent vectors and the discriminative information

        # try some different architectures here to see which one works the best, try different network depths and adding some normalization layers

        self.h_scalar = nn.Sequential( # the scalar function for the unconditioned part
        nn.Linear(encoder_output_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
        )

        self.l_proj = nn.Sequential( # the scalar function for the unconditioned part
        nn.Linear(encoder_output_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 128)) # projector from the latent vector to the common space shared with the dsicriminative information
        
        self.d_proj = nn.Linear(dis_info_dim, proj_dim) # projector from the discriminative information space to the common space shared with the the latent vectors


        self.optimizer = Adam(itertools.chain(self.h_scalar.parameters(), self.l_proj.parameters(), self.d_proj.parameters()), 
                              lr=lr,
                              weight_decay=weight_decay)
        self.scheduler = ExponentialLR(optimizer=self.optimizer, gamma=1)
    
    def forward(self, l, dis_info):

        # feed the latent vector to the scalar function
        uncond_logit = self.h_scalar(l)

        # projecting to the common space
        l_projected = self.l_proj(l)
        d_projected = self.d_proj(dis_info)

        # calculate the dot product
        cond_logit = torch.sum(l_projected * d_projected, dim=1, keepdim=True)

        logit = uncond_logit + cond_logit

        return (logit, ) # return the logit value computed by the discriminator
    
    def freeze(self):
        logger.info(f"freeze the {self.model_name}")
        for p in self.h_scalar.parameters():
            p.requires_grad = False
        for p in self.l_proj.parameters():
            p.requires_grad = False
        for p in self.d_proj.parameters():
            p.requires_grad = False
    
    def unfreeze(self):
        logger.info(f"unfreeze the {self.model_name}")
        for p in self.h_scalar.parameters():
            p.requires_grad = True
        for p in self.l_proj.parameters():
            p.requires_grad = True
        for p in self.d_proj.parameters():
            p.requires_grad = True

class VisionProjConditionAdversarialAC(VisionConditionAdversarialAdaptAC):

    feature_fields = ['camera', 'velocity', 'state', 'curvature']
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
        self.actor = VisionConditionalAdversarialActor(pretrain_agent=pretrain_agent)

        # initialize the discriminator, the discriminator here is the projection discriminator
        self.discriminator = ProjDiscriminator(lr = ad_agent_params['lr_discriminator'], 
                                           weight_decay = ad_agent_params['weight_decay'],
                                           encoder_output_dim = ad_agent_params["encoder_output_dim"],
                                           dis_info_dim = ad_agent_params['dis_info_dim'])
        
        self.actor.set_discriminator(self.discriminator)

        self.actor.to(ptu.device)
        self.discriminator.to(ptu.device)

class VisionNullAdaptAC(VisionAdversarialAdaptAC):

    feature_fields = ['camera', 'velocity']
    label_fields = ['action', "domain_indicator"]

    def __init__(self, pretrain_agent : VisionNaiveRandomization, pretrain_agent_params: dict, ad_agent_params: dict):
        """

        This is the most fundamental baseline adaptor.
        No discriminator is involved in the domain transfer process. The only data available is the pretraining data from
        the source domains.

        @ pretrain_encoder_path: the path to load the weight of the pretrained agent (the agent that will be adapted to the new environment)
        @ pretrain_agent_params: the parameters of the pretrained encoder
        @ ad_agent_params: the parameters of the discriminators, and some relevant modifications that should be made to the pretrain_agent params
        Model Input: states
        Model Output: actions

        Loss: MSE
        """
        super().__init__(pretrain_agent = None, pretrain_agent_params = None, ad_agent_params = None)
        self.actor = VisionAdversarialActor(pretrain_agent=pretrain_agent, adv_factor = 0.0) # set adv_factor to disable the discriminator

        # initialize the discriminator
        # initialize the discriminator
        self.discriminator = Discriminator(lr = ad_agent_params['lr_discriminator'], 
                                           weight_decay = ad_agent_params['weight_decay'],
                                           encoder_output_dim = ad_agent_params["encoder_output_dim"],
                                           dis_info_dim = ad_agent_params['dis_info_dim'])
        
        self.actor.set_discriminator(self.discriminator)

        self.actor.to(ptu.device)
        self.discriminator.to(ptu.device)
    
    def fit(self, train_dataset: sourceTargetBalanceBuffer, n_epochs, val_dataset=None, global_step=None):
        info = defaultdict(lambda: {})
        # first train the discriminator
        logger.info(f"The training for the discriminator is skipped .....")

        #train the actor
        logger.info(f"///// Training the actor ///// [{self.actor.model_name}]")
        actor_info = self.actor.fit(train_dataset=train_dataset, n_epochs=n_epochs)

        for d in (actor_info, ):
            for k1, v1 in d.items():
                for k2, v2 in v1.items():
                    info[k1][k2] = v2
        return info