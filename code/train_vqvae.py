import sys
import os
# to avoid module not found error when VQVAE imports Encoder class
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/models/')

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from models.VQVAE import VQVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current device: {device}\n", flush=True)

# directory setup
MAIN_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../'
if 'generated_images' not in os.listdir(MAIN_DIR):
    print('creating generated_images directory...', flush=True)
    os.mkdir(MAIN_DIR + 'generated_images')
if 'logs' not in os.listdir(MAIN_DIR):
    print('creating logs directory...', flush=True)
    os.mkdir(MAIN_DIR + 'logs')
if 'reconstructed_images' not in os.listdir(MAIN_DIR):
    print('creating reconstructed_images directory...', flush=True)
    os.mkdir(MAIN_DIR + 'reconstructed_images')
if 'c' not in os.listdir(MAIN_DIR):
    print('creating saved_models directory...', flush=True)
    os.mkdir(MAIN_DIR + 'saved_models')