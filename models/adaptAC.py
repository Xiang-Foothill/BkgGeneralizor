from models.visionSafeAC import Discriminator, VisionAdversarialActor, VisionAdversarialAdaptAC, VisionNaiveRandomization, VisionConditionalAdversarialActor
import torch
from utils import pytorch_util as ptu
from loguru import logger

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