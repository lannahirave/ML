import random
import os
import torch
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast

from models import CaptioningModel
import matplotlib.pyplot as plt


def visualize_samples(data_loader: DataLoader, num_samples: int = 6):
    cols = int(num_samples**0.5)
    rows = (num_samples + cols - 1) // cols

    plt.figure(figsize=(15, 30))

    for i in range(num_samples):
        image, caption = data_loader.dataset[
            random.randint(0, len(data_loader.dataset) - 1)
        ]
        plt.subplot(rows, cols, i + 1)
        plt.imshow(image)
        plt.title(caption.replace("<|endoftext|>", ""))
        plt.axis("off")
    if not os.path.exists("results"):
        os.makedirs("results")
    plt.savefig("results/samples.png")
    plt.show()

