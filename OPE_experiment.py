"""verify the correlation between the latent space conditional dicriminator loss and the agent's online performance in the target domain"""
from il_estimator import il_Estimator, generate_target_buffer
from il_CAD_BARC_trainer import BARC4_PID, BARC5_PID, PRETRAIN_BRIGHT1, PRETRAIN_BRIGHT2, PRETRAIN_BRIGHT3
from il_NR_trainer import DOMAIN1, DOMAIN2, DOMAIN5, DOMAIN6, DOMAIN7, DOMAIN8, DOMAIN9, DOMAIN10, DOMAIN11, DOMAIN12, DOMAIN13, DOMAIN14, IL_Trainer_CARLA_VisionNaiveRandomizationAC
import gym
import gym_carla
from loguru import logger
import yaml
from pathlib import Path
from models.visionSafeAC import VisionNaiveMultihead, VisionNaiveRandomization, VisionCompleteMultiHead, VisionAdversarialAdaptAC, VisionAdversarialActor, VisionConditionAdversarialAdaptAC, VisionNullAdaptAC, VisionProjConditionAdversarialAC
from utils.data_util import EfficientReplayBuffer
import numpy as np
from src.carla_gym.controllers.barc_pid import PIDWrapper
import utils.plot_util as pul
import torch
OPE_AGENT0 = "OPE_test_agent0"
OPE_AGENT1 = "OPE_test_agent1"
OPE_AGENT2 = "OPE_test_agent2"
OPE_AGENT3 = "OPE_test_agent3"
OPE_AGENT4 = "OPE_test_agent4"
OPE_AGENT5 = "OPE_test_agent5"
OPE_AGENT6 = "OPE_test_agent6"
OPE_AGENT7 = "OPE_test_agent7"
OPE_AGENT8 = "OPE_test_agent8"
OPE_AGENT9 = "OPE_test_agent9"
OPE_AGENT10 = "OPE_test_agent10"
OPE_AGENT11 = "OPE_test_agent11"
OPE_AGENT12 = "OPE_test_agent12"
OPE_AGENT13 = "OPE_test_agent13"
OPE_AGENT14 = "OPE_test_agent14"
OPE_AGENT15 = "OPE_test_agent15"
OPE_AGENT16 = "OPE_test_agent16"
OPE_AGENT17 = "OPE_test_agent17"

def train_agents(domain_list = [DOMAIN1, DOMAIN2, DOMAIN6, DOMAIN7, DOMAIN8, DOMAIN9, DOMAIN10, DOMAIN11, DOMAIN12, DOMAIN13, DOMAIN14], random_seed = 55):

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    """train a list of agents for the correlation test"""
    max_epochs = 10 # the maximum epochs for one-agent training
    # set the common params for il_NR_trainer
    t0, dt, dt_sim = 0., 0.1, 0.01
    carla_params = dict(
        track_name='L_track_barc',
        t0=t0, dt=dt, dt_sim=dt_sim,
        do_render=False,
        max_n_laps=50,
        enable_camera=True,
        host='localhost',
        port=2000,
    )
    env = gym.make('barc-v0', **carla_params)

    #load the agent params
    config_path = "./config/VisionAttentionActor.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    agent_params = config['model_hparams']

    il_trainer_params = dict(
        carla_params = carla_params,
        expert_cls = PIDWrapper,
        augment_percent=0.6,
        initial_traj_len=1024,
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
        beta=0.4,
        pretrain_critic=False,
        beta_decay_freq=1,
        save_profile = True,
        to_reload = False,
        latent = False,
        mid_freeze = np.inf,
        to_PCA = False,
        save_data = True,
        env = env,
        **agent_params
    )
    
    for i, domain in enumerate(domain_list):
        # iterate through the domain_list and train one agent for each of these domains
        il_trainer_params['domain_list'] = [domain] # single-element domain list
        il_trainer_params["comment"] = 'OPE_test_agent' + str(i + 9)
        trainer = IL_Trainer_CARLA_VisionNaiveRandomizationAC(**il_trainer_params)
        trainer.main(n_epochs=max_epochs)

def correlation_test():
    # define the target domain
    eval_domain = DOMAIN11
    map_name = eval_domain["map_name"]
    weatherID = eval_domain["weatherID"]
    t0, dt, dt_sim = 0., 0.1, 0.01
    carla_params = dict(
        track_name='L_track_barc',
        t0=t0, dt=dt, dt_sim=dt_sim,
        do_render=False,
        max_n_laps=50,
        enable_camera=True,
        host='localhost',
        port=2000,
        weatherID = weatherID,
        map_name = map_name
    )
    env = gym.make('barc-v0', **carla_params)
    agents = [OPE_AGENT0, OPE_AGENT1, OPE_AGENT3, 
              OPE_AGENT4, OPE_AGENT5, OPE_AGENT6, 
              OPE_AGENT7, OPE_AGENT8, OPE_AGENT9,
              OPE_AGENT10, OPE_AGENT11, OPE_AGENT12, 
              OPE_AGENT13, OPE_AGENT14, OPE_AGENT15, 
              OPE_AGENT16, OPE_AGENT17, BARC4_PID, BARC5_PID] # the agents to be estimated
    # agents = [OPE_AGENT0, OPE_AGENT1, OPE_AGENT8]

    # The discriminator_params configuration
    config_path = "./config/VisionCAD.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    discriminator_params = config['model_hparams']
    #collect the target domain data
    target_buffer = generate_target_buffer(env = env)
    discriminator_params['target_buffer'] = target_buffer

    res = {} # the result dictionary
    for agent_path in agents:
        agent = load_agent(path = agent_path)
        
        #load the source domain data of the pretrained model
        source_buffer = load_source_buffer(agent_path)
        discriminator_params['source_buffer'] = source_buffer

        estimator = il_Estimator(env = env, carla_params = carla_params, agent = agent, **discriminator_params)
        data = estimator.main()
        res[agent_path] = data

        logger.debug(res[agent_path])

        # save the result
        np.savez_compressed("graphs/raw_data/OPE_raw_data_only_x_tran.npz", **res)
        logger.info(f"OPE estimation done for agent {agent_path}!")
    
    return res

def load_agent(path):
    pretrain_config_path = "./config/VisionAttentionActor.yaml"
    with open(pretrain_config_path, 'r') as f:
        config = yaml.safe_load(f)

    pretrain_agent_params = config['model_hparams']

    # modify the learning rate of the pretrained agent for better fine tuning
    pretrain_agent_params['lr'] = pretrain_agent_params['lr']
    pretrain_agent_params['gamma'] = 1.0 # no weight decay during fine tuning
    pretrain_agent = VisionNaiveRandomization(**pretrain_agent_params)

    logger.info("//// Loading pretrained encoders ////")
    pretrain_agent.load(path=Path(__file__).resolve().parent / 'model_data',
                        name=path)
    return pretrain_agent

def load_source_buffer(agent_path):
    """load the source-domain data buffer"""
    source_buffer = EfficientReplayBuffer(maxsize=19000, # hardcode the source_buffer's maxsize
                                        lazy_init= True)
    data_dir = 'data'
    source_buffer.load(path = data_dir, name = agent_path)

    return source_buffer

def load_raw_data(data_path = 'graphs/raw_data/OPE_raw_data.npz'):
    loaded = np.load(data_path, allow_pickle=True)
    data_dict = {key: loaded[key].item() for key  in loaded.files}
    return data_dict

if __name__ == '__main__':
    # eval_data = correlation_test()
    eval_data = load_raw_data()
    pul.plot_OPE_eval(data = eval_data, kl_thresholds=[1.0, 2.0], variance_type="error_bar")
    # train_agents()
    