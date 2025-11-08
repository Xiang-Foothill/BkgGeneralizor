"""An agent estimator:
In the init function specify the following:
1. the agent to be estimated
2. the estimation metrics to be done"""
from models.visionSafeAC import VisionNaiveMultihead, VisionNaiveRandomization, VisionCompleteMultiHead, VisionAdversarialAdaptAC, VisionAdversarialActor, VisionConditionAdversarialAdaptAC, VisionNullAdaptAC, VisionProjConditionAdversarialAC
import gym
import carla_gym

class il_Estimator():

    def __init__(self,
                env,
                carla_params,
                agent : VisionNaiveRandomization,
                metrics : list,
                **discriminator_params):
        
        """agent: the agent to be estiamted
        metrics: a list of strings, each representing an estimation function"""

        metrics_dict = { # Key: metrics name, Value: an estimation function

        }
        
        self.agent = agent

        # make a carla environment is the environment is not passed in
        if env is None:
            self.env = gym.make('barc-v0', **carla_params)
        else:
            self.env = env

        raise NotImplementedError
    
    def main():
        raise NotImplementedError
