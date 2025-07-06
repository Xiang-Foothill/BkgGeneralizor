from models.visionSafeAC import Discriminator, VisionAdversarialActor, VisionAdversarialAdaptAC, VisionNaiveRandomization, VisionConditionalAdversarialActor, VisionConditionAdversarialAdaptAC, CatDiscriminator
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
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

class WassersteinDiscriminator(Discriminator):
    """Wasserstein discriminator with soft gradient penalty and explicit gradient norms."""

    def __init__(self, lr, weight_decay, encoder_output_dim, dis_info_dim, null_init=False, gp_lambda=0.5):
        super().__init__(lr, weight_decay, encoder_output_dim, dis_info_dim, null_init)
        self.gp_lambda = gp_lambda # parameter for gradient penalty

    def forward(self, l):
        # l: latent vectors, requires_grad=True
        l = l.clone().detach().requires_grad_(True)
        domain_logits = self.D(l)  # critic score

        # Compute gradient norms w.r.t. inputs
        grad_outputs = torch.ones_like(domain_logits)
        gradients = torch.autograd.grad(
            outputs=domain_logits,
            inputs=l,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        grad_norms = gradients.view(gradients.size(0), -1).norm(2, dim=1).detach()

        return (domain_logits, grad_norms)

    def loss(self, pred, label):
        domain_logits, grad_norms = pred
        domain_ind, = label

        source_mask = domain_ind == 1
        target_mask = domain_ind == 0

        source_score = domain_logits[source_mask].mean()
        target_score = domain_logits[target_mask].mean()
        wasserstein_distance = source_score - target_score
        loss_d = -wasserstein_distance

        # Soft gradient penalty: penalize ||∇D(l)||₂ − 1
        gradient_penalty = ((grad_norms - 1.0) ** 2).mean()
        loss_d += self.gp_lambda * gradient_penalty

        return loss_d, {
            'wasserstein_distance': wasserstein_distance.item(),
            'discriminator_loss': loss_d.item(),
            'gradient_penalty': gradient_penalty.item()
        }

class CatWassersteinDiscriminator(WassersteinDiscriminator):
    feature_fields = ['camera', 'state', 'curvature']
    label_fields = ['domain_indicator']

    def input_buffer(self, features, actor : VisionAdversarialActor ):
        """preprocess the features before passing them to the model.
        note that in the dataloader, there are only RGB images, there is no such a field called latent vector"""

        img, state, curvature = features
        dis_info = self.discriminative_info(state, curvature)

        latent_vectors = actor.get_latent(img = img, to_numpy = False)
        latent_vectors = latent_vectors.detach() # cut from the rest of the computation graph

        return [latent_vectors, dis_info]
    
    def discriminative_info(self, state, curvature):
        """extract task relevant information from the state variable and the curvature variable,
        which is discriminative in all the domains"""
        task_info_state = state[:, -2:]        # shape: [batch_size, 2]
        task_info_curvature = curvature        # shape: [batch_size, 3]
        return torch.cat([task_info_state, task_info_curvature], dim=1)  # shape: [batch_size, 5]
    
    def forward(self, l, dis_info):

        """The conditioning policy is simply concatenate the latent vectors with the discriminative information"""

        combined = torch.cat([l, dis_info], dim = 1)
        return super().forward(combined) # return the logit value computed by the discriminator

class WassersteinAdversarialActor(VisionAdversarialActor):
    def loss(self, pred, label):
        u_pred, domain_logits = pred

        u, domain_ind = label
        domain_ind = domain_ind.view(-1, 1)

        # === Source and target masks ===
        source_mask = domain_ind.bool().squeeze(1)
        target_mask = ~source_mask

        # === Policy loss: supervised imitation loss on source ===
        if source_mask.sum() > 0:
            policy_loss = self.loss_func(u_pred[source_mask], u[source_mask])
        else:
            policy_loss = torch.tensor(0.0, device=u.device)

        # === Adversarial loss: maximize Wasserstein distance ===
        if source_mask.sum() > 0 and target_mask.sum() > 0:
            source_score = domain_logits[source_mask].mean()
            target_score = domain_logits[target_mask].mean()
            wasserstein_distance = source_score - target_score
            adv_loss = wasserstein_distance
        else:
            adv_loss = torch.tensor(0.0, device=u.device)

        total_loss = policy_loss + self.adv_factor * adv_loss

        return total_loss, {
            'policy_loss': policy_loss.item(),
            'adversarial_loss': adv_loss.item(),
            'total_loss': total_loss.item()
        }

class WassersteinConditionAdversarialActor(WassersteinAdversarialActor):

    feature_fields = ['camera', 'velocity', 'state', 'curvature']
    label_fields = ['action', "domain_indicator"]

    def forward(self, img, vel, state, curvature):

        # Normalize the image first. 
        img = img.permute(0, 3, 1, 2) / 255.

        l = self.resnet(img)
        dis_info = self.discriminator.discriminative_info(state, curvature)

        # logger.info(f"the disinfo = {dis_info}")
        
        outputs = self.discriminator(l, dis_info)  # allow gradients to flow

        domain_logits = outputs[0]

        if self.check_latent_collapse(l):
            logger.info("WARNING: latent space collapse is detected!")

        v_encoded = self.velocity_encoder(vel)
        combined = torch.cat([l, v_encoded], dim=1)  # shape: [batch_size, latent_dim]

        output = self.decision(combined)
        return (output, domain_logits)

class WassersteinAdversarialAdaptAC(VisionAdversarialAdaptAC):

    def __init__(self, pretrain_agent : VisionNaiveRandomization, pretrain_agent_params: dict, ad_agent_params: dict):
        super().__init__(pretrain_agent = None, pretrain_agent_params = None, ad_agent_params = None) # null init the super class
        self.actor = WassersteinAdversarialActor(pretrain_agent=pretrain_agent)

        # initialize the discriminator
        self.discriminator = WassersteinDiscriminator(lr = ad_agent_params['lr_discriminator'], 
                                           weight_decay = ad_agent_params['weight_decay'],
                                           encoder_output_dim = ad_agent_params["encoder_output_dim"],
                                           dis_info_dim = ad_agent_params['dis_info_dim'])
        
        self.actor.set_discriminator(self.discriminator)

        self.actor.to(ptu.device)
        self.discriminator.to(ptu.device)

class WassersteinConditionAdversarialAdaptAC(VisionAdversarialAdaptAC):

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

        if pretrain_agent == None:
            return # the null initialization
        
        self.actor = WassersteinConditionAdversarialActor(pretrain_agent=pretrain_agent)

        # initialize the discriminator
        self.discriminator = CatWassersteinDiscriminator(lr = ad_agent_params['lr_discriminator'], 
                                           weight_decay = ad_agent_params['weight_decay'],
                                           encoder_output_dim = ad_agent_params["encoder_output_dim"],
                                           dis_info_dim = ad_agent_params['dis_info_dim'])
        
        self.actor.set_discriminator(self.discriminator)

        self.actor.to(ptu.device)
        self.discriminator.to(ptu.device)

class DensityEstimator(BaseModel):
    
    def __init__(self):
        super().__init__()
        self.has_fit = False
        self.kde_src : KernelDensity
        self.kde_tgt : KernelDensity

    def dis_info_collate(self, dataset: EfficientReplayBuffer):
        state = dataset.fields["state"][:, -2:]
        curvature = dataset.fields["states"]
        dis_info = np.concatenate(state, curvature)
        logger.debug(dis_info)
        return dis_info
    
    def best_bandwidth(self, data):
        # Choose a range of candidate bandwidths
        bandwidths = np.logspace(-1.5, 1, 20)

        grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                            {'bandwidth': bandwidths},
                            cv=5)  # 5-fold cross-validation

        grid.fit(data)  # X = your dataset (source or target)
        best_h = grid.best_params_['bandwidth']

        return best_h

    def fit(self, dataset: sourceTargetBalanceBuffer):
        if self.has_fit:
            logger.warning("The Density Ratio Estimator should only be fit for one time!")
        
        logger.info("Fitting the density ratio estimator...")
        dis_info_source = self.dis_info_collate(dataset.source_buffer)
        dis_info_target = self.dis_info_collate(dataset.target_buffer)

        source_bandwidth = self.best_bandwidth(dis_info_source)
        target_bandwidth = self.best_bandwidth(dis_info_target)
        logger.info(f"source domain bandwidth = {source_bandwidth}; target domain bandwidth = {target_bandwidth}")

        self.kde_src = KernelDensity(kernel='gaussian', bandwidth=source_bandwidth).fit(dis_info_source)
        self.kde_tgt = KernelDensity(kernel='gaussian', bandwidth=target_bandwidth).fit(dis_info_target)
        logger.info(f"The ratio estimator is now fit.")
        self.has_fit = True
    
    def densities(self, dis_info):

        if isinstance(dis_info, torch.Tensor):
            dis_info = ptu.clone_to_numpy()

        source_est = self.kde_src.score_samples(dis_info)
        target_est = self.kde_tgt.score_samples(dis_info)

        logger.debug(source_est)
        return source_est, target_est
    
    def sourceTargetRatios(self, dis_info, epsilon: float = 1e-8, normalize: bool = True):
        """
        Compute density ratios w(x) = p_source(x) / (p_target(x) + epsilon)
        
        Args:
            dis_info (np.ndarray or torch.Tensor): Input features of shape [B, D]
            epsilon (float): Small value to avoid division by zero (log-domain)
            normalize (bool): Whether to normalize the output weights to have mean 1

        Returns:
            np.ndarray: Density ratios of shape [B]
        """
        if isinstance(dis_info, torch.Tensor):
            dis_info = ptu.clone_to_numpy(dis_info)

        log_p_src, log_p_tgt = self.densities(dis_info)

        # Stabilize denominator by clipping log_p_tgt
        log_p_tgt_clipped = np.maximum(log_p_tgt, np.log(epsilon))

        # Compute log ratio, then exponentiate
        log_ratio = log_p_src - log_p_tgt_clipped
        ratio = np.exp(log_ratio)

        # Optional normalization to have mean 1
        if normalize:
            ratio /= np.mean(ratio)

        return ratio

class VisionConditionalAdversarialReweightActor(VisionConditionalAdversarialActor):
    """Based on the VisionConditionalAdversarialActor, when calculating adversarial loss, each target buffer data point will be weighted based on
    KDE esimation of discriminative information probability in both domains"""

    def set_density_estimator(self, density_estimator : DensityEstimator):
        """set the KDE estimators as class attributes"""
        self.density_estimator = density_estimator

    def forward(self, img, vel, state, curvature):

        # Normalize the image first. 
        img = img.permute(0, 3, 1, 2) / 255.

        l = self.resnet(img)
        dis_info = self.discriminator.discriminative_info(state, curvature)

        # logger.info(f"the disinfo = {dis_info}")
        
        outputs = self.discriminator(l, dis_info)  # allow gradients to flow

        domain_logits = outputs[0]

        if self.check_latent_collapse(l):
            logger.info("WARNING: latent space collapse is detected!")

        v_encoded = self.velocity_encoder(vel)
        combined = torch.cat([l, v_encoded], dim=1)  # shape: [batch_size, latent_dim]

        output = self.decision(combined)
        return (output, domain_logits)
    
    def loss(self, pred, label):
        raise NotImplementedError
        u_pred, domain_logits = pred # the u_pred here has a shape of [64, 2], [batch_size, output_size]
        u, domain_ind = label

        domain_ind = domain_ind.view(-1, 1)  # Ensure consistent shape: [batch_size, 1]
        #    === Source mask for policy loss ===
        source_mask = domain_ind.bool().squeeze(1)  # shape: [batch_size]

        # === Policy loss: only for source samples ===
        if source_mask.sum() > 0:
            policy_loss = self.loss_func(u_pred[source_mask], u[source_mask])
        else:
            policy_loss = torch.tensor(0.0, device=u.device)

        # Adversarial loss: encourage encoder to fool discriminator
        # Flip domain labels: try to make domain_ind_pred look like the opposite (e.g., 0.5 or source)
        # target_labels = 1. - domain_ind.float()  # invert labels
        target_labels = torch.ones_like(domain_ind, dtype=torch.float32) * 0.9  # new uniform label: 0.9 for all

        adv_loss_fn = nn.BCEWithLogitsLoss()
        adv_loss = adv_loss_fn(domain_logits, target_labels)

        total_loss = policy_loss + self.adv_factor * adv_loss
        return total_loss, {
                'policy_loss': policy_loss.item(),
                'adversarial_loss': adv_loss.item(),
                'total_loss': total_loss.item()
        }

class VisionConditionAdversarialReweightAdaptAC(VisionConditionAdversarialAdaptAC):

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
        super().__init__(pretrain_agent = pretrain_agent, pretrain_agent_params = pretrain_agent_params, ad_agent_params = ad_agent_params)
        self.density_estimator = DensityEstimator()
    
    def fit(self, train_dataset: sourceTargetBalanceBuffer, n_epochs, val_dataset=None, global_step=None):
        if global_step == 0:
            self.density_estimator.fit(train_dataset) # only fit the estimator at the first epoch

        info = defaultdict(lambda: {})
        # first train the discriminator
        logger.info(f"///// Training the discriminator [{self.discriminator.model_name}] /////")
        discriminator_info = self.discriminator.fit(n_epochs = n_epochs * 3, train_dataset = train_dataset, actor = self.actor)

        #train the actor
        logger.info(f"///// Training the actor ///// [{self.actor.model_name}]")
        actor_info = self.actor.fit(train_dataset=train_dataset, n_epochs = n_epochs)

        for d in (discriminator_info, actor_info):
            for k1, v1 in d.items():
                for k2, v2 in v1.items():
                    info[k1][k2] = v2
        return info