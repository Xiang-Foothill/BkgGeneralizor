import os
import sys

import numpy as np
from matplotlib import pyplot as plt
from huggingface_hub import login
# import cv2

from mpclab_common.track import get_track
from mpclab_common.pytypes import VehicleState
import pickle
import carla
from pathlib import Path
from diffusers import IFInpaintingPipeline
import torch
import random
from PIL import Image

TOKEN = "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"
DEVICE = "cuda"
PATH = Path(__file__).parent / 'prompt_embeds_dict.pth'
SEMNATICS_SELECTED = [11]
PROMPTS = [
    # "race-car stadium"
    "coastal roads",
    "urban streets of modern cities",
]
INFERENCE = 2  # 4

""" Available text_prompts, choose one from below as the text-prompt for image generation:

    'cloudy sky',
    'raining sky',
    'sunny sky',
    'high quality photo',
    'road under the sky',
    'urban streets of modern cities',
    "snow mountains",
    "desert",
    "dense forests",
    "tropical forests",
    "race-car stadium",
    "urban tunnels",
    "rural farmland",
    "coastal roads",
    ""

    """
class fusion_camera:
    def __init__(self, track_name, dictionary_path = PATH,  semantics_selected = SEMNATICS_SELECTED, prompts = PROMPTS, inference_steps = INFERENCE):
        print("""////////// Initiation of fusion_camera starts ////////////""")

        self.semantics_selected = semantics_selected
        self.prompts = prompts
        self.inference_steps = inference_steps

        login(TOKEN)
        self.prompt_embeds_dict = torch.load(dictionary_path)
        print( f"----- prompt_embeds ditionary successfully loaded with keys {list(self.prompt_embeds_dict.keys())}----- ")

        self.client = carla.Client('192.168.50.51', 2000)
        self.obs_size = 256 # 64 # DO NOT change the image size here!!!!! The inpainter object is very sensitive to the input image
        self.dt = 0.1

        self.rgb_img = np.empty((self.obs_size, self.obs_size, 3), dtype=np.uint8)
        self.semantic_mask = np.empty((self.obs_size, self.obs_size, 3), dtype=np.uint8)
        self.inpaint_img = np.empty((self.obs_size, self.obs_size, 3), dtype=np.uint8)

        self.world = None
        self.camera_bp = None
        self.track_obj = get_track(track_name)

        # so far we are developing the test script that can be run without loading opendrive_map, get back this line later
        self.load_opendrive_map(track_name)
        # self.test_init()

        # Temporary: built-in rendering
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2)
        self.im1, self.im2 = self.ax1.imshow(self.rgb_img), self.ax2.imshow(self.inpaint_img)

        self.spawn_rgb_camera()
        self.spawn_semantic_camera()
        print("----- cameras successfully loaded -----")
        
        self.inpainter = IFInpaintingPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", 
                                                       variant="fp16", 
                                                       torch_dtype=torch.float16, 
                                                       text_encoder = None)
        self.inpainter.to(DEVICE)
        print("----- Deepfloyd Model successfully loaded -----")

    @property
    def height(self):
        return self.rgb_img.shape[0]
    
    @property
    def width(self):
        return self.rgb_img.shape[1]
    
    def test_init(self):
        self.world = self.client.get_world()# get rid of this line later when finishing devleoping
        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = True
        self.settings.fixed_delta_seconds = self.dt
        self.world.apply_settings(self.settings)

    def load_opendrive_map(self, track_name):
        xodr_path = Path(__file__).resolve().parents[1] / 'OpenDrive' / f"{track_name}.xodr"
        if not os.path.exists(xodr_path):
            raise ValueError("The file does not exist.")
            return
    
        with open(xodr_path, encoding='utf-8') as od_file:
            try:
                data = od_file.read()
            except OSError:
                print('file could not be read.')
                sys.exit()
        print('load opendrive map %r.' % os.path.basename(xodr_path))
        vertex_distance = 2.0  # in meters
        max_road_length = 0.1  # in meters
        wall_height = 0.2      # in meters
        extra_width = 0.1       # in meters
        self.world = self.client.generate_opendrive_world(
                        data, carla.OpendriveGenerationParameters(
                            vertex_distance=vertex_distance,
                            max_road_length=max_road_length,
                            wall_height=wall_height,
                            additional_width=extra_width,
                            smooth_junctions=True,
                            enable_mesh_visibility=True))
        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = True
        self.settings.fixed_delta_seconds = self.dt
        self.world.apply_settings(self.settings)

        # Next, try to spawn a camera at the origin. 
    def spawn_rgb_camera(self):
        self.rgb_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.rgb_bp.set_attribute('image_size_x', str(self.obs_size))
        self.rgb_bp.set_attribute('image_size_y', str(self.obs_size))
        self.rgb_bp.set_attribute('fov', '110')
        # Set the time in seconds between sensor captures
        self.rgb_bp.set_attribute('sensor_tick', f"{self.dt}")

        def get_rgb_img(data):
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.rgb_img = array

        self.camera_trans = carla.Transform(carla.Location(x=0, y=0, z=0.2))
        self.rgb_sensor = self.world.spawn_actor(self.rgb_bp, self.camera_trans)
        self.rgb_sensor.listen(get_rgb_img)
    
    def spawn_semantic_camera(self):
        self.semantic_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        self.semantic_bp.set_attribute('image_size_x', str(self.obs_size))
        self.semantic_bp.set_attribute('image_size_y', str(self.obs_size))
        self.semantic_bp.set_attribute('fov', '110')
        # Set the time in seconds between sensor captures
        self.semantic_bp.set_attribute('sensor_tick', f"{self.dt}")

        def mask_generator(data):
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]

            array = array.copy()
            array.flags.writeable = True
            array[:, :, 1] = array[:, :, 0]
            array[:, :, 2] = array[:, :, 0]
            semantic_tags = array

            self.semantic_mask = np.logical_or.reduce([semantic_tags == semantic for semantic in self.semantics_selected])

        self.camera_trans = carla.Transform(carla.Location(x=0, y=0, z=0.2))
        self.semantic_sensor = self.world.spawn_actor(self.semantic_bp, self.camera_trans)
        self.semantic_sensor.listen(mask_generator)
        
    def query_rgb(self, state):

        self.rgb_sensor.set_transform(carla.Transform(carla.Location(x=state.x.x, y=-state.x.y, z=0.2), 
                                                         carla.Rotation(yaw=-np.rad2deg(state.e.psi))))
        self.semantic_sensor.set_transform(carla.Transform(carla.Location(x=state.x.x, y=-state.x.y, z=0.2), 
                                                         carla.Rotation(yaw=-np.rad2deg(state.e.psi))))
        self.world.tick()

        self.inpaint()

        self.im1.set_array(self.rgb_img)
        self.im2.set_array(self.inpaint_img)

        plt.pause(0.5)
    
        return self.inpaint_img
    
    def query_data(self, state):

        self.rgb_sensor.set_transform(carla.Transform(carla.Location(x=state.x.x, y=-state.x.y, z=1.0), 
                                                         carla.Rotation(yaw=-np.rad2deg(state.e.psi))))
        self.semantic_sensor.set_transform(carla.Transform(carla.Location(x=state.x.x, y=-state.x.y, z=1.0), 
                                                         carla.Rotation(yaw=-np.rad2deg(state.e.psi))))
        self.world.tick()

        self.inpaint()

        self.im1.set_array(self.rgb_img)
        self.im2.set_array(self.inpaint_img)

        plt.pause(0.01)
    
        return self.inpaint_img, self.rgb_img
    
    def inpaint(self):

        def IFBuffer(original_image, original_mask):
            def downsample(img):
                return np.array(Image.fromarray(img, mode='RGB').resize((64, 64)))
            original_image = downsample(original_image)
            original_mask = downsample(original_mask)
            original_image = original_image.astype(np.float32)
            original_mask = original_mask.astype(np.float32)

            image = torch.tensor(original_image, dtype = torch.float32)
            mask_image = torch.tensor(original_mask, dtype = torch.float32)

            image = (image / 255.0) * 2 - 1
            image = image.permute(2, 0, 1)
            mask_image = mask_image.permute(2, 0, 1)

            return image.unsqueeze(0), mask_image.unsqueeze(0) # use the unsqueeze function to add a batch dimension
        
        image, mask_image = IFBuffer(self.rgb_img, self.semantic_mask)

        # prompt_embeds = torch.cat([
        #     self.prompt_embeds_dict[prompt] for prompt in self.prompts
        #     ], dim=0)
        # negative_prompt_embeds = torch.cat(
        #     [self.prompt_embeds_dict['']] * len(self.prompts)
        #             )
        prompt_embeds = self.prompt_embeds_dict[random.choice(self.prompts)]
        negative_prompt_embeds = self.prompt_embeds_dict['']
        # generator = torch.manual_seed(1)

        inpaint_output = self.inpainter(
                num_inference_steps=self.inference_steps,
                image=image.half(),
                mask_image=mask_image.half(),
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                # generator=generator,
                output_type="pt",
            ).images
        
        inpaint_output = inpaint_output[0].cpu()
        self.inpaint_img = ((inpaint_output.permute(1, 2, 0) / 2.0 + 0.5).detach().numpy() * 255.0).astype(np.uint8)

        return self.inpaint_img

    def test_data_collect(self):
        data = {}
        spawn_points = self.world.get_map().get_spawn_points()
        semantics_selected = [11]
        data["rgb"] = np.zeros(shape = (len(spawn_points), self.height, self.width, 3), dtype = np.uint8)
        data["mask"] = np.zeros(shape = (len(spawn_points), self.height, self.width, 3), dtype = np.uint8)

        for i, spawn_point in enumerate(spawn_points):
            state = VehicleState()
            state.x.x, state.x.y, state.e.psi = spawn_point.location.x, - spawn_point.location.y, np.random.uniform(low = 0.0, high = 2* np.pi)

            self.query_data(state, semantics_selected)
            data["rgb"][i] = self.rgb_img
            data["mask"][i] = self.semantic_mask
        
        # fig, ax = plt.subplots()
        # im = ax.imshow(np.empty((256, 256, 3), dtype=np.uint8))

        # for i in range(len(data["rgb"])):
        #     temp_image = data["rgb"][i]
        #     temp_mask = data["mask"][i]
        #     im.set_array(temp_image)
        #     plt.pause(0.5)

        # data["rgb"] = np.asarray(data["rgb"])
        # data["mask"] = np.asarray(data["mask"])

        with open("test_data.pkl", "wb") as f:
                pickle.dump(data, f)

def check_test_data():
    with open("test_data.pkl", "rb") as file:
        data = pickle.load(file)
    fig, ax = plt.subplots()
    im = ax.imshow(np.empty((256, 256, 3), dtype=np.uint8))
    print(data["rgb"].shape)
    for i in range(data["rgb"].shape[0] - 1):
        temp_image = data["rgb"][i]
        temp_mask = data["mask"][i + 1]
        im.set_array(temp_image * temp_mask)
        plt.pause(0.5)

def carla_test():
    camera = fusion_camera(track_name='L_track_barc')
    spawn_points = camera.world.get_map().get_spawn_points()

    for i, spawn_point in enumerate(spawn_points):
        state = VehicleState()
        state.x.x, state.x.y, state.e.psi = spawn_point.location.x, - spawn_point.location.y, np.random.uniform(low = 0.0, high = 2* np.pi)
        camera.query_data(state)
    
if __name__ == "__main__":
    # carla_test()
    carla_test()
