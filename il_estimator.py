"""An agent estimator:
In the init function specify the following:
1. the agent to be estimated
2. the estimation metrics to be done"""
import numpy as np
from models.visionSafeAC import VisionNaiveMultihead, VisionNaiveRandomization, VisionCompleteMultiHead, VisionAdversarialAdaptAC, VisionAdversarialActor, VisionConditionAdversarialAdaptAC, VisionNullAdaptAC, VisionProjConditionAdversarialAC, CatDiscriminator
import gym
import gym_carla
from src.carla_gym.controllers.barc_mpcc_conv import MPCCConvWrapper
from src.carla_gym.controllers.barc_pid import PIDWrapper
from loguru import logger
import utils.pytorch_util as ptu
from utils.data_util import EfficientReplayBuffer, sourceTargetBalanceBuffer, to_dataloader
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import utils.plot_util as plu
import torch

expert_mp = {
    'pid': PIDWrapper,
    'mpcc-conv': MPCCConvWrapper,
}

class il_Estimator():

    def __init__(self,
                env,
                carla_params,
                agent : VisionNaiveRandomization,
                online_metrics : list = None,
                **discriminator_params):
        
        """agent: the agent to be estiamted
        metrics: a list of strings, each representing an estimation function"""
        self.carla_params = carla_params
        self.online_metrics = online_metrics
        self.discriminator_params = discriminator_params

        self.agent = agent
        self.agent.to(ptu.device)

        # make a carla environment is the environment is not passed in
        #TODO: we need to specify the domain via carla_params
        if env is None:
            self.env = gym.make('barc-v0', **carla_params)
        else:
            self.env = env
    
    def online_il_loss(self, expert : str = 'pid', sample_num : int = 5):
        """The imitation loss of the agent along its on-policy trajectory in the target domain.
        @expert: the expert with which the imitation loss is calculated against, specify the exper with a string
        @sample_num: the number of trajectories to be sampled
        @return: average imitation loss along the agent's trajectory in the target domain"""

        #initiate the expert
        expert_cls = expert_mp[expert]
        self.expert = expert_cls(dt=self.carla_params['dt'], t0=self.carla_params['t0'], track_obj=self.env.get_track())

        max_traj_len = 800 # the maximum laps for a single sampled trajectory
        self.agent.reset()
        self.agent.eval() # prepare the agent for evaluation

        data = {
            "agent_actions" : [],
            "expert_actions" : []
        }

        for _ in range(sample_num):
            ob, info = self.env.reset(options={'controller': self.expert, 'spawning': 'fixed'}) # reset the environment
            traj_len = 0
            agent_actions = []
            expert_actions = []

            while traj_len <= max_traj_len:
                ac = self.agent.get_action(*self.agent.parse_carla_obs(ob, info))
                expert_ac, expert_info = self.expert.step(**ob, **info)
                expert_ac = np.clip(expert_ac, self.env.action_space.low, self.env.action_space.high)

                agent_actions.append(ac)
                expert_actions.append(expert_ac)

                closed_loop_action = ac # use the agent action as the closed loop action
                next_ob, rew, terminated, truncated, info = self.env.step(closed_loop_action) # step the environment

                ob = next_ob
                traj_len += 1

                if truncated:
                    break
            
            data["agent_actions"].append(np.asarray(agent_actions))
            data['expert_actions'].append(np.asarray(expert_actions))

        def to_metrics(data):
            """Compute online imitation loss (MSE) and avg trajectory length."""
            agent_trajs = [np.asarray(a) for a in data["agent_actions"]]
            expert_trajs = [np.asarray(e) for e in data["expert_actions"]]
            lengths = [min(len(a), len(e)) for a, e in zip(agent_trajs, expert_trajs) if len(a) and len(e)]
            if not lengths:
                return {'online_imitation_loss': np.nan, 'average_traj_lens': 0.0}

            diffs = np.concatenate([a[:L] - e[:L] for a, e, L in zip(agent_trajs, expert_trajs, lengths)], axis=0)
            loss = np.mean(np.square(diffs))
            return {'online_imitation_loss': float(loss), 'average_traj_lens': float(np.mean(lengths))}

        metrics = to_metrics(data)

        return data, metrics
    
    def discriminator_KL(self, perct_list = [0.7, 0.2, 0.1]):
        """estimate the conditional KL divergence via the discriminator
        perct_list = [train_percentage, val_percentage, infer_percentage]
        - train_percentage of data will be used to train the discriminator
        - val_percentage is used as validation set to judge early stop
        - infer_percentage is used to estimate the KL divergence"""

        if abs(sum(perct_list) - 1.0) > 1e-4:
            raise ValueError("The summation of percentage list must be 1.0")
        train_percent, val_percent, infer_percent = perct_list

        discriminator = CatDiscriminator(lr = self.discriminator_params['lr_discriminator'], 
                                           weight_decay = self.discriminator_params['weight_decay'],
                                           encoder_output_dim = self.discriminator_params["encoder_output_dim"],
                                           dis_info_mode = self.discriminator_params['dis_info_mode'])
        discriminator.to(ptu.device)
        # prepare the data for discriminator training
        try:
            source_buffer = self.discriminator_params['source_buffer']
        except KeyError:
            raise KeyError("The source buffer needs to be passed in for discriminator loss estimation")
        try:
            logger.info("The target buffer is passed in...")
            target_buffer = self.discriminator_params['target_buffer']
        except KeyError:
            logger.info("The target buffer is not passed in. Now recollecting data from the target domain.")
            target_buffer = generate_target_buffer(env = self.env)

        source_total, target_total = len(source_buffer), len(target_buffer)
        source_val_len, target_val_len = int(source_total * val_percent), int(target_total * val_percent)
        source_train_len, target_train_len = int(source_total * train_percent), int(target_total * train_percent)  # ensure total sums up exactly
        source_infer_len, target_infer_len = source_total - (source_val_len + source_train_len), target_total - (target_val_len + target_train_len)

        #split the data
        source_train_buffer, source_val_buffer, source_infer_buffer = random_split(source_buffer, [source_train_len, source_val_len, source_infer_len])
        target_train_buffer, target_val_buffer, target_infer_buffer = random_split(target_buffer, [target_train_len, target_val_len, target_infer_len])
        # logger.debug(f"source_train_len = {len(source_train_buffer)}); source_val_len = {len(source_val_buffer)}); target_train_len = {len(target_train_buffer)}); target_val_len = {len(target_val_buffer)});")
        train_buffer = sourceTargetBalanceBuffer(source_buffer = source_train_buffer, target_buffer = target_train_buffer)
        val_buffer = sourceTargetBalanceBuffer(source_buffer = source_val_buffer, target_buffer = target_val_buffer)
        discriminator.fit_ES(train_dataset = train_buffer, val_dataset = val_buffer, n_epochs=20, actor = self.agent, debug = False)

        logger.debug(f"the source infer buffer has size of {len(source_infer_buffer)}")

        #estimate the KL divergence using the trained discriminator
        discriminator.eval()
        self.agent.eval()
        logits = []

        with torch.no_grad():
            infer_loader = to_dataloader(source_infer_buffer, batch_size=64, shuffle=False, num_workers=0,
                                            manifest=[discriminator.feature_fields,
                                                      discriminator.label_fields])
            for features, labels in infer_loader:
                latent_features = discriminator.input_buffer(features, actor=self.agent)
                pred = discriminator(*latent_features)
                batch_logits = ptu.to_numpy(pred[0]) # output logits of the discriminator
                logits.append(batch_logits)
        
        logits = np.concatenate(logits, axis = 0)
        source_probs = 1. / (1. + np.exp(- logits))
        target_probs = 1 - source_probs
        KL_est = np.mean(np.log(source_probs) - np.log(target_probs))
        return KL_est
    
    def discriminator_KL_group(self, perct_list = [0.7, 0.2, 0.1], eval_num = 5):
        """call the discriminator_KL estimation function for eval_num times,
        return the results as a numpy list"""
        res = []
        for _ in range(eval_num):
            res.append(self.discriminator_KL(perct_list))
        return np.asarray(res)

    def main(self):
        data, metrics = self.online_il_loss(expert = 'pid')
        data['KL_divergence estimation'] = self.discriminator_KL_group()
        return data

def generate_target_buffer(env, buffer_size =2048, sample_distribution = 'naive_random'):
        """A helper function that generates the target_domain data"""
        target_buffer = EfficientReplayBuffer(maxsize=19000, # hardcode the source_buffer's maxsize
                                        lazy_init= True)
        ob, info = env.reset(options={})
        traj_len = 0

        with tqdm(total=buffer_size, desc='Sampling', unit='steps') as pbar:
            while traj_len < buffer_size:
                domain_indicator = np.asarray([0.]) # the target domain has domain label 0.0
                ob['domain_indicator'] = domain_indicator

                if sample_distribution == 'naive_random':
                    next_ob = env.naive_random_obs()
                elif sample_distribution == 'first_4m_random':
                    next_ob = env.first4mDistribution_random_obs() # choose the random sample distribution policy
                elif sample_distribution == 'middle_3m_random':
                    next_ob = env.middle3mDistribution_random_obs()
                else:
                    raise ValueError(f"the sample_distribution {sample_distribution} is not available")

                #some null observation without specific meaning but only for data structure consistency
                rew, expert_ac, closed_loop_action = 0.0, np.zeros((2,)), np.zeros((2,))

                # Store all info necessary for add_frame in a dict
                frame = {
                    'ob': ob,
                    'rew': rew,
                    'terminated': False,
                    'truncated' : False,
                    'info': info,
                    'action': expert_ac.astype(np.float32),
                    'closed_loop_action': closed_loop_action.astype(np.float32),
                    'next_state': next_ob['state'],
                }

                target_buffer.add_frame(
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
        
        return target_buffer
