from huggingface_hub import login
from transformers import T5EncoderModel
from diffusers import DiffusionPipeline
import torch

TOKEN = "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"
PATH = 'prompt_embeds_dict.pth'

DEVICE = "cuda"

class prompt_genrator():
    """To save the usage of cuda memory when running fusion_camera, the text prompt_embeds are generated inde-
    pendently by this prompt_generator and saved as a dictionary
    How to use this generator:

    1. Initiate a new generator object 
    generator = pompt_camera(dctionary_path)

    2. add the text prompts that you want to encode and save in the dictionary
    generator.add(**Text prompts** here)
    
    This generator does not rely on cuda usage, the text-encoder pipline is run on CPU instead of GPU"""
    
    def __init__(self, dictionary_path = PATH):
        """@dictionary_path: the directory where the dictionary for the original text-prompt dictionary is stored
        If such a directory does not exist, a new dicionary together with a new directory will be created"""
        login(TOKEN) # login to huggingface
        self.path = dictionary_path

        # Load the T5 text encoder (this may take a while)
        text_encoder = T5EncoderModel.from_pretrained(
            "DeepFloyd/IF-I-L-v1.0",
            subfolder="text_encoder",
            variant="8bit",
            )
        self.text_pipe = DiffusionPipeline.from_pretrained(
            "DeepFloyd/IF-I-L-v1.0",
            text_encoder=text_encoder,  # pass the previously instantiated text encoder
            unet=None
            )

        try:
            self.prompt_embeds_dict = torch.load(self.path)
            print(f"Dictionary loaded, it has keys below: {list(self.prompt_embeds_dict.keys())}")
        except:
            self.prompt_embeds_dict = {}
            print("////////// A new dictionary is created //////////")
        
    def add_prompts(self, *args):
        for new_prompt in args:
            if new_prompt in self.prompt_embeds_dict:
                print(f"Prompt embeds {new_prompt}  already exists !!!!!")
            else:
                prompt_embeds, _ = self.text_pipe.encode_prompt(new_prompt)
                self.prompt_embeds_dict[new_prompt] = prompt_embeds
                print(f"prompt {new_prompt} encoded and saved !!!!!")
        
        torch.save(self.prompt_embeds_dict, self.path)

if __name__ == "__main__":
    generator = prompt_genrator(PATH)
    generator.add_prompts("text, letters, watermark, logo, signature, captions, subtitles, words, ")