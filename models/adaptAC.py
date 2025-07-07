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
import matplotlib.pyplot as plt

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
        self.kde_src : KernelDensity = None
        self.kde_tgt : KernelDensity = None

    def dis_info_collate(self, dataset: EfficientReplayBuffer):
        state = dataset.retrieve_entire_field("state")[:, -2:]
        curvature = dataset.retrieve_entire_field("curvature")
        dis_info = np.concatenate([state, curvature], axis = 1)
        return dis_info
    
    def best_bandwidth(self, data, start: float, end: float):
        # Choose a range of candidate bandwidths
        bandwidths = np.linspace(start, end, 10)

        grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                            {'bandwidth': bandwidths},
                            cv=3)  # 3-fold cross-validation

        grid.fit(data)  # X = your dataset (source or target)
        best_h = grid.best_params_['bandwidth']

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

        # source_bandwidth = self.best_bandwidth(dis_info_source, start = 0.02, end = 0.05)
        # target_bandwidth = self.best_bandwidth(dis_info_target, start = 0.055, end = 0.08)

        source_bandwidth = 0.04 # the empirical value for the dataset of pretrain_bright3
        # target_bandwidth = 0.15 # the empirical value for the target domain dataset collected with the naive random sampling policy
        target_bandwidth = 0.055 # the empirical value for first4mDistribution with a size of 512

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

        st_ratios = self.density_estimator.sourceTargetRatios(dis_info, to_tensor = True) # the source target ratios
        outputs = self.discriminator(l, dis_info)  # allow gradients to flow

        domain_logits = outputs[0]

        if self.check_latent_collapse(l):
            logger.info("WARNING: latent space collapse is detected!")

        v_encoded = self.velocity_encoder(vel)
        combined = torch.cat([l, v_encoded], dim=1)  # shape: [batch_size, latent_dim]

        output = self.decision(combined)
        return (output, domain_logits, st_ratios)
    
    def loss(self, pred, label):
        u_pred, domain_logits, st_ratios = pred  # [B, 2], [B, 1], [B]
        u, domain_ind = label                    # u: [B, 2], domain_ind: [B]

        domain_ind = domain_ind.view(-1, 1)  # [B, 1]
        source_mask = domain_ind.bool().squeeze(1)   # True if source
        target_mask = ~source_mask                  # True if target

        # === Policy loss: only source samples ===
        if source_mask.sum() > 0:
            policy_loss = self.loss_func(u_pred[source_mask], u[source_mask])
        else:
            policy_loss = torch.tensor(0.0, device=u.device)

        # === Adversarial loss ===
        target_labels = torch.ones_like(domain_ind, dtype=torch.float32) * 0.9  # encourage confusion

        # Compute per-sample BCE loss
        adv_loss_raw = nn.functional.binary_cross_entropy_with_logits(
            domain_logits, target_labels, reduction='none'  # [B, 1]
        ).squeeze(1)  # shape: [B]

        st_ratios = st_ratios.detach().to(dtype=adv_loss_raw.dtype, device=adv_loss_raw.device)  # stop gradients through density ratio

        # Only weight the loss for target-domain samples
        adv_loss = torch.zeros_like(adv_loss_raw)
        adv_loss[target_mask] = adv_loss_raw[target_mask] * st_ratios[target_mask]
        adv_loss[source_mask] = adv_loss_raw[source_mask]  # optionally: keep as is or mask to 0

        # Normalize (optional): prevent magnitude explosion
        adv_loss = adv_loss.mean()

        # === Total loss ===
        total_loss = policy_loss + self.adv_factor * adv_loss

        return total_loss, {
            'policy_loss': policy_loss.item(),
            'adversarial_loss': adv_loss.item(),
            'total_loss': total_loss.item()
        }

class CatReweightDiscriminator(CatDiscriminator):
    def set_density_estimator(self, density_estimator : DensityEstimator):
        """set the KDE estimators as class attributes"""
        self.density_estimator = density_estimator
    
    def forward(self, l, dis_info):

        """The conditioning policy is simply concatenate the latent vectors with the discriminative information"""

        combined = torch.cat([l, dis_info], dim = 1)
        st_ratios = self.density_estimator.sourceTargetRatios(dis_info)

        return (self.D(combined), st_ratios) # return the logit value computed by the discriminator
    
    def loss(self, pred, label):
        domain_logits, st_ratios = pred  # domain_logits: [B, 1], st_ratios: [B]
        domain_ind, = label              # domain_ind: [B, 1] (1 for source, 0 for target)

        domain_ind = domain_ind.view(-1, 1)
        source_mask = domain_ind.bool().squeeze(1)
        target_mask = ~source_mask

        # Compute per-sample BCE loss
        bce_raw = nn.functional.binary_cross_entropy_with_logits(
            domain_logits, domain_ind.float(), reduction='none'
        ).squeeze(1)  # shape: [B]

        st_ratios = st_ratios.detach().to(dtype=bce_raw.dtype, device=bce_raw.device)  # no gradient back into KDE

        # Apply weights
        weighted_loss = torch.zeros_like(bce_raw)
        weighted_loss[source_mask] = bce_raw[source_mask]  # optionally leave unweighted
        weighted_loss[target_mask] = bce_raw[target_mask] * st_ratios[target_mask]

        loss_d = weighted_loss.mean()

        return loss_d, {'discriminator_loss': loss_d.item()}

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
        super().__init__(pretrain_agent = None, pretrain_agent_params = None, ad_agent_params = None)

        if pretrain_agent == None:
            return # the null initialization
        
        self.actor = VisionConditionalAdversarialReweightActor(pretrain_agent=pretrain_agent)

        # initialize the discriminator
        # initialize the discriminator
        self.discriminator = CatReweightDiscriminator(lr = ad_agent_params['lr_discriminator'], 
                                           weight_decay = ad_agent_params['weight_decay'],
                                           encoder_output_dim = ad_agent_params["encoder_output_dim"],
                                           dis_info_dim = ad_agent_params['dis_info_dim'])
        
        self.actor.set_discriminator(self.discriminator)
        self.density_estimator = DensityEstimator()

        self.actor.to(ptu.device)
        self.discriminator.to(ptu.device)

        self.actor.set_density_estimator(self.density_estimator)
        self.discriminator.set_density_estimator(self.density_estimator)
    
    def fit(self, train_dataset: sourceTargetBalanceBuffer, n_epochs, val_dataset=None, global_step=None):
        if global_step == 0:
            self.density_estimator.fit(train_dataset, display_ratio=True) # only fit the estimator at the first epoch

        info = defaultdict(lambda: {})
        # first train the discriminator
        logger.info(f"///// Training the discriminator [{self.discriminator.model_name}] /////")
        discriminator_info = self.discriminator.fit(n_epochs = n_epochs * 4, train_dataset = train_dataset, actor = self.actor)

        #train the actor
        logger.info(f"///// Training the actor ///// [{self.actor.model_name}]")
        actor_info = self.actor.fit(train_dataset=train_dataset, n_epochs = n_epochs)

        for d in (discriminator_info, actor_info):
            for k1, v1 in d.items():
                for k2, v2 in v1.items():
                    info[k1][k2] = v2
        return info