import matplotlib.pyplot as plt
import time
import torch
import numpy as np

def load_images(path="Bkgset.pth"):
    data = torch.load(path)  # expects a torch tensor of shape [N, H, W, 3]
    return data

def play_images_like_anime(images, duration=1.0, n_images=50):
    """
    Displays images one after another in the same figure window.
    
    Args:
        images (numpy.ndarray): [N, H, W, 3] array of images.
        duration (float): Time in seconds each image is displayed.
        n_images (int): Number of images to show.
    """
    n_images = min(n_images, len(images))
    fig, ax = plt.subplots()
    im = ax.imshow(images[0].astype(np.uint8))
    ax.axis("off")

    for i in range(n_images):
        im.set_data(images[i].astype(np.uint8))
        ax.set_title(f"Image {i}")
        plt.pause(duration)

    plt.close()

images = load_images("Bkgset.pth")
play_images_like_anime(images, duration=1.0, n_images=100)
