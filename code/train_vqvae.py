import sys
import os
# to avoid module not found error when VQVAE imports Encoder class
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/models/')

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
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
if 'saved_models' not in os.listdir(MAIN_DIR):
    print('creating saved_models directory...', flush=True)
    os.mkdir(MAIN_DIR + 'saved_models')

# 1. load the data
# reference: https://github.com/jpowie01/DCGAN_CelebA/blob/master/dataset.py
ROOT_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../data/'
BATCH_SIZE = 1

train_transform = transforms.Compose([
    # make image h == image w
    # NOTE: original size = (218, 178)
    transforms.CenterCrop((178, 178)),
    # if the image is resized immediately without making h == w, the image can look distorted
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = datasets.ImageFolder(root=ROOT_DIR, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

print(f"""
Total data: {len(trainset)}
""", flush=True)