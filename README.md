# TCADT Implementation For Vision-based Imitation Learning
This repository contains the implementation for *Task-Conditional Adversarial Learning: A Generalizable Framework for Off-policy Unsupervised Vision Domain Transfer*. We base the implementation on vision-based imitation learning for path tracking task in carla. In this repo, we will see how an imitation learning agent can be transfered to some drastically different visual domains. 

## Prerequisites

### Python Environment Setup
We recommend using **Python 3.8**, as this implementation has only been tested on it.

```sh
conda create -n CAIL python=3.8
conda activate CAIL
pip install -r requirements.txt
pip install -e src/carla_gym/gym-carla
pip install -e src/mpclab_common
pip install -e src/mpclab_controllers
pip install -e src/mpclab_simulation
```

### CARLA Installation
This implementation relies on **CARLA** for camera-based experiments.

- Follow the [CARLA official installation guide](https://carla.readthedocs.io/en/latest/start_quickstart/) for your OS. The simplest way is to download and unzip the precompiled version.
- We used **CARLA 0.9.15** in our simulations. You can download it [here](https://github.com/carla-simulator/carla/releases/tag/0.9.15).

#### Installing CARLA's Python API
For proper simulation functionality, install CARLA's Python API based on your Python version:

- **Python 3.8 (Recommended)**: Use the provided `.whl` file.
  ```sh
  pip install dist/carla-0.9.15-cp38-cp38-linux_x86_64.whl
  ```
- **Python 3.7**: If using the precompiled version, run the following after unzipping:
  ```sh
  cd $CARLA_ROOT/PythonAPI/carla/dist
  pip install carla-0.9.15-cp37-cp37m-manylinux_2_27_x86_64.whl
  ```
  Replace `$CARLA_ROOT` with the CARLA installation directory.
- **Other Python Versions**: You must build CARLA from source and generate the Python API accordingly. Follow the [official build instructions](https://carla.readthedocs.io/en/latest/build_system/).

### Domain Configruation
Each visual domain is define as a combinatory of a Carla weather ID and a Carla map. See `config/domain_config.yaml` to see how a domain is defined.
In our experiments, we use RoadRunner to customize a list of Carla maps, and the domain configuration is defined based on these customize Carla maps. Please go to [packaged carla maps](https://drive.google.com/drive/folders/14teonEY2k9htesbKSOkvE1HP_VTtuUa9?usp=sharing) to download the packaged versions of these customized maps. 
As a shortcut for packages download, you can run the following from the root directory of your carla folder
```
pip install gdown
cd Import
gdown --folder "https://drive.google.com/drive/folders/14teonEY2k9htesbKSOkvE1HP_VTtuUa9?usp=sharing" -O temp_dl
mv temp_dl/* .
rm -r temp_dl
```
To import these packaged maps into 
a packaged version of Carla, please place all the downloaded `.tar.gz` files under the `CARLA_ROOT/Import` directory of your Carla simulator, and run the following from the root directory
of your Carla simulator:

```
./ImportAssets.sh
```

You can also create your own visual domain by adding a dictionary in `config/domain_config.yaml` in the following format:

```
DOMAIN6 [The dmain reference used by il_TCADT_trainer.py command line]:
  name: "bus stops before sunset" [The name of the domain]
  map_name: *L_TRACK_BARC6 [the name of an available map from your Carla simulation]
  weatherID: 4 [the weather id for Carla]
```

### HPIPM Installation
If you want to run the experiment with mpcc-conv expert, please install **HPIPM** as the optimization solver.

Follow the [HPIPM installation guide](https://github.com/giaf/hpipm) to set up the environment and install its Python API in your Python environment.

---

## Running the Experiments
Due to the hyper-parameter inrobustness of adverarial learning families, we highly recommend any interested future researchers to carefully tune the learning rates for actors and discriminators:
- Configure the model hyperparameters in:
  - `config/VisionCAD.yaml` (for TCADT actor and discriminator)
- When running experiment with Carla, we recommend boosting the carla server with the low-quality command for better reliability:
  ```sh
  ./CarlaUE4.sh -RenderOffScreen -quality-level=Low
  ```

### Running Specific Experiments
Please run the file `il_TCADT_trainer.py` to conduct the visual-domain transfer experiment for vision-based end-to-end imitation learning. The default exper is an PID controller.

#### Example Command Lines
Listed below are some command line examples for calling `il_TCADT_trainer.py`
- Use `DOMAIN4 DOMAIN5` as source domains. Use `demo1` as pretrained agent and transfer it to `DOMAIN11` while using `state_curvature` as the discriminative information. Follow the the distribution `naive_random` to collect the target domain buffer. When the domain transfer is done, save the model with name `TCADT1`.
```
python il_TCADT_trainer.py --source_domains DOMAIN4 DOMAIN5 --target_domains DOMAIN11 -p demo1 -d state_curvature -ts naive_random -m TCADT1
```

- Use `DOMAIN1 DOMAIN2` as source domains. Use `demo1` as pretrained agent and transfer it to `DOMAIN11` while using the default discriminative information. Follow the the distribution `naive_random` to collect the target domain buffer with a size of 248. When the domain transfer is done, save the model with name `TCADT1`.
```
python il_TCADT_trainer.py --source_domains DOMAIN1 DOMAIN2 --target_domains DOMAIN11 -p demo1 -t 248  -ts naive_random -m TCADT1
```

#### Siginificant Arguments
Listed below are some significant arguments that can be passed in as command line for `il_TCADT_trainer.py`

- **`--source_domains` (`-sd`)**  
  A list of source-domain names from which the agent can access *unrestricted* amounts of expert demonstrations and training data. These domains are assumed to be well-understood, fully supervised, and serve as the foundation for pretraining or multi-source domain generalization.

- **`--target_domains` (`-td`)**  
  A list of target-domain names representing the deployment environments. Only a *limited, offline, reward-free* dataset is collected from each target domain. During training, the agent uses these samples for adversarial alignment, OPE estimation, or contrastive adaptation.

- **`--pretrain_agent` (`-p`)**  
  Specifies the name of a pretrained agent to load before domain adaptation.   
  - Use `"null"` to train a completely new agent from scratch without loading any pretrained parameters.
  - We prepared two pretrained agents available for visual domain transfer: `demo1` and `demo2`.

- **`--target_sample_distribution` (`-ts`)**  
  Determines the sampling distribution used when collecting offline data in the target domain. This controls *where* along the track or trajectory the frames are sampled, which affects the distribution shift between source and target datasets.  
  - **`naive_random`**: Uniform random sampling over the entire target-domain trajectory.  
  - **`first_4m_random`**: Randomly samples with a majority from the first 4 meters (useful for examining localized domain shift).  
  - **`middle_3m_random`**: Random Samples with a majority from middle section of the track to study mid-track domain discrepancies.

- **`--dis_info_mode` (`-d`)**  
  Specifies the type of *discriminative information* \(y\) used for latent alignment and adversarial training. This variable affects the conditional distribution \(p(l \mid y)\) and determines how strongly the discriminator can distinguish domains.  
  Available modes:
  - **`only_curvature`**: Uses only the road curvature signal.
  - **`state_curvature`**: Uses both the vehicle state (lateral transition and heading angle) and road curvature; typically the most informative setting.
  - **`only_state`**: Uses only state variables without curvature; useful when geometric cues should be excluded.
  - **`only_x_tran`**: Uses only lateral translation; studies alignment under a single scalar discriminative variable.
  - **`gps_xy`**: Uses 2D GPS position in the global frame; provides spatial alignment based on global location.
  - **`gps_full`**: Uses full GPS state (global coordiante, heading angle, and velocity).

- **`--target_domain_len` (`-t`)**  
  Specifies the total number of target-domain samples to collect when building the offline target buffer.  
  This value directly controls the *size* of the target-domain dataset used for adaptation, OPE estimation, and discriminator training.  
  A larger value provides more coverage of the target environment (reducing variance in KL estimation), while a smaller value simulates a more challenging low-data adaptation regime.  
  The default value `2048` corresponds to collecting approximately one short trajectory's worth of observations in the target domain.
  
#### Pretrain Agent

We highly recommend pretraining the agent using the **DAgger pipeline with domain randomization**, as this provides a strong and stable initialization before performing domain adaptation.

To pretrain an agent, run:

```bash
python il_NR_trainer.py -td [list of source domains for pretraining] -m [pretrained_agent_name]

```
One example:
```
python il_NR_trainer.py -td DOMAIN2 DOMAIN7 -m demo3
```
Note that the name of the pretrained agent can be directly passed in as `--pretrain_agent` `-pt` argument for `il_TCADT_trainer.py`. 

### Additional Help
For a full list of available command-line arguments, run:
```
python il_TCADT_trainer.py --help
```
