from diffusers import DiffusionPipeline
import torch
import random
import numpy as np
from huggingface_hub import login
import pathlib as Path
import matplotlib.pyplot as plt
import os

device = "cuda"

TOKEN = "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"
DICT_PATH =  os.path.dirname(os.path.abspath(__file__)) + '\prompt_embeds_dict.pth'
MIX_PATH =  os.path.dirname(os.path.abspath(__file__)) + "/fractal_mixset.pth"
DEVICE = "cuda"
DISPLAY_NUM = 20 # number of generated images displayed
DISPLAY_GEN = True # whether to display the generated images or not
GEN_NUM = 50 # the number of images to be generated
PROMPTS = [
     "recursive fractal",
    "chaotic fractal",
    "symmetrical fractal",
    ]

""" Available text_prompts, choose from below as the text-prompt for image generation:

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
    "recursive fractal",
    "chaotic fractal",
    "symmetrical fractal",
    ""

    """

class mixsetGenerator():
    """This generator is used for adding pictures to the mixset

    @dictionary_path: the directory in which you stored your encoded text_prompts. If such a path does not exist, error will be raised.
    @mixset_path: the directory in which you stored your mixset images. If such a path does not exist, a new path will be created

    1. Initiate a new generator object 
    generator = mixsetGenerator(dctionary_path, mixset_path)

    2. add the text prompts that you want to encode and save in the dictionary
    generator.add(number of images, [**Text prompts** here]) if the text_prompts given is None, prompts will be randomly selected from the 
    prompts in the text_emhbeds dictionary

    The images will be stored in the set as a numpy array with shape[N, H, W, 3] with format of np.uint8
    """
    
    def __init__(self, dictionary_path = DICT_PATH, mixset_path = MIX_PATH):
        """@dictionary_path: the directory where the dictionary for the original text-prompt dictionary is stored
        If such a directory does not exist, a new dicionary together with a new directory will be created"""
        login(TOKEN) # login to huggingface
        self.dictionary_path = dictionary_path
        self.mixset_path = mixset_path

        # Load the T5 text encoder (this may take a while)
        # stage 1
        self.model = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-M-v1.0", variant="fp16", torch_dtype=torch.float16, text_encoder = None)
        self.model.to(DEVICE)

        try:
            self.prompt_embeds_dict = torch.load(self.dictionary_path)
            print(f"/////////////////////////// Dictionary loaded, it has keys below. please choose prompts from the keys given: {list(self.prompt_embeds_dict.keys())} ////////////////////////////////////////////")
        except:
            print(f"/////////////////////////// No text_embeds dictionary found in {dictionary_path}. Please make sure you have an encoded text-prompts dictionary stored before you use this generator ////////////////////////////////////////////////")
        
        try:
            self.mixset = [torch.load(self.mixset_path)]
            print(f"//////////////////////////////////////////// Mixset loaded, it already has {self.mixset[0].shape[0]} pictures //////////////////////////////")
        except:
            self.mixset = []
            print(f"////////// A new mixset is created in the directory{self.mixset_path} //////////")

    def add_images(self, n, prompts, display_generated = True):

        def generate_embeds():
            prompt_embeds = []
            negative_prompt_embeds = []

            prompt_keys = list(self.prompt_embeds_dict.keys()) if prompts is None else prompts
            for prompt in prompt_keys:
                if prompt not in self.prompt_embeds_dict:
                    print(f"the prompt given {prompt} cannot be found in the text-embeds dictionary, please double check the keys")
            if '' in prompt_keys:
                prompt_keys.remove('')

            for i in range(n):
                prompt = random.choice(prompt_keys)
                prompt_embeds.append(self.prompt_embeds_dict[prompt])
                negative_prompt_embeds.append(self.prompt_embeds_dict[''])

            prompt_embeds = torch.cat(prompt_embeds)
            negative_prompt_embeds = torch.cat(negative_prompt_embeds)

            return prompt_embeds, negative_prompt_embeds
        
        prompt_embeds, negative_embeds = generate_embeds()
        images = self.model(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, output_type="pt", num_inference_steps=20).images
        add_set = (((images.permute(0, 2, 3, 1).cpu().detach().numpy()) / 2.0 + 0.5) * 255.0).astype(np.uint8)

        if display_generated:
            fig, axes = plt.subplots(DISPLAY_NUM // 5 + 1, 5)
            for i in range(DISPLAY_NUM):
                picked_index = random.choice(list(range(add_set.shape[0])))
                axes[i // 5, i % 5].imshow(add_set[picked_index])
            
            plt.show()

            while True:
                user_input = input("--------------------------Do you want to add the generated image set to the existing directory? enter Y/N:").strip().lower()
                if user_input == 'y':
                    break
                if user_input == "n":
                    raise Exception("The gnerated mixset is discarded")
            
        self.mixset.append(add_set)
        torch.save(np.concatenate(self.mixset), self.mixset_path)
        print(f"Generated mixset added to the directory{self.mixset_path}")

if __name__ == "__main__":
    generator = mixsetGenerator()
    generator.add_images(GEN_NUM, PROMPTS)