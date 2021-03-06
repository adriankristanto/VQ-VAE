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

DATASET = 'mnist'

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current device: {device}\n", flush=True)

    # directory setup
    MAIN_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../'
    if 'data' not in os.listdir(MAIN_DIR):
        print('creating data directory...', flush=True)
        os.mkdir(MAIN_DIR + 'data')
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
    BATCH_SIZE = 128
    NUM_WORKERS = 2

    train_transform = None
    trainset = None
    in_channels = None

    if DATASET == 'celeba':
        in_channels = 3

        train_transform = transforms.Compose([
            # make image h == image w
            # NOTE: original size = (218, 178)
            transforms.CenterCrop((178, 178)),
            # if the image is resized immediately without making h == w, the image can look distorted
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        trainset = datasets.ImageFolder(root=ROOT_DIR + 'CelebA', transform=train_transform)

    elif DATASET == 'mnist':
        in_channels = 1

        train_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        trainset = datasets.MNIST(root=ROOT_DIR, download=True, transform=train_transform)
    
    elif DATASET == 'cifar10':
        in_channels = 3

        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        trainset = datasets.CIFAR10(root=ROOT_DIR, train=True, transform=train_transform, download=True)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    print(f"""
    Total data: {len(trainset)}
    """, flush=True)

    # 2. instantiate the model
    net = VQVAE(
        in_channels=in_channels,
        hidden_channels=128,
        num_resblocks=2,
        res_channels=32,
        D=64,
        K=512,
        beta=0.25,
        gamma=0.99
    )

    print(f"{net}\n", flush=True)

    multigpu = False
    if torch.cuda.device_count() > 1:
        print(f'Number of GPUs: {torch.cuda.device_count()}\n', flush=True)
        net = nn.DataParallel(net)
        multigpu = True

    net.to(device)

    # 3. define the loss function
    # this is the first loss term, which will be optimized by both the encoder and the decoder
    # the third loss term, which will be optimized by the encoder, will be returned by the vq layer
    # the second loss term will not be used here, as EMA update is used.
    reconstruction_loss = nn.MSELoss()

    # 4. define the optimiser
    # the learning rate used in the original implementation by the author
    LEARNING_RATE = 3e-4
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    # 5. train the model
    MODEL_DIRPATH = os.path.dirname(os.path.realpath(__file__)) + '/../saved_models/'
    RECONSTRUCTED_DIRPATH = os.path.dirname(os.path.realpath(__file__)) + '/../reconstructed_images/'
    CONTINUE_TRAIN = False
    CONTINUE_TRAIN_NAME = MODEL_DIRPATH + 'vqvae-model-epoch10.pth'
    EPOCH = 200
    SAVE_INTERVAL = 10
    # for reconstruction test
    RECONSTRUCTION_SIZE = 64

    def save_training_progress(epoch, net, optimizer, filename):
        torch.save({
            # since the currect epoch has been completed, save the next epoch
            'epoch' : epoch + 1,
            'net_state_dict' : net.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
        }, filename)

    next_epoch = 0
    if CONTINUE_TRAIN:
        checkpoint = torch.load(CONTINUE_TRAIN_NAME)
        net.load_state_dict(checkpoint.get('net_state_dict'))
        optimizer.load_state_dict(checkpoint.get('optimizer_state_dict'))
        next_epoch = checkpoint.get('epoch')

    # training loop
    for epoch in range(next_epoch, EPOCH):
        # running_loss = 0.0

        trainloader = tqdm(trainloader)

        net.train()
        for i, train_data in enumerate(trainloader):
            inputs = train_data[0].to(device)
            # 1. zeroes the gradients
            # optimizer.zero_grad() vs net.zero_grad()
            # reference: https://discuss.pytorch.org/t/model-zero-grad-or-optimizer-zero-grad/28426
            net.zero_grad()
            # 2. forward propagation
            outputs, commitment_loss, _, _ = net(inputs)
            # 3. compute loss
            # note: the commitment loss has been multiplied by beta in the vq layer
            if multigpu:
                # if we are using multiple gpus for training, there will be x number of loss returned as a tensor 
                # instead of just a scalar
                # where x is the number of gpus that we used for training
                # this can cause an issue with the autograd
                # therefore, we need to make it a scalar by taking the mean
                commitment_loss = torch.mean(commitment_loss)
            mse = reconstruction_loss(outputs, inputs)
            loss = mse + commitment_loss
            # 4. backward propagation
            loss.backward()
            # 5. update parameters
            optimizer.step()

            # running_loss += loss.item()

            trainloader.set_description((
                f"epoch: {epoch+1}/{EPOCH}; "
                f"mse: {mse.item():.5f}; "
                f"latent: {commitment_loss.item():.5f}; "
                f"loss: {loss.item():.5f}"
            ))

            if i % 469 == 0:
                net.eval()
                sample = inputs[:RECONSTRUCTION_SIZE]
                with torch.no_grad():
                    outputs, _, _, _ = net(sample)
                torchvision.utils.save_image(
                    torch.cat([sample, outputs], dim=0),
                    RECONSTRUCTED_DIRPATH + f"vqvae_{epoch+1}_{i}.png",
                    normalize=True,
                    range=(-1, 1)
                )
        
        # save the model
        if (epoch + 1) % SAVE_INTERVAL == 0:
            save_training_progress(epoch, net, optimizer, MODEL_DIRPATH + f'vqvae-model-epoch{epoch + 1}.pth')
        
        # print(f"Training loss: {running_loss / len(trainloader)}", flush=True)

    # save the model
    save_training_progress(epoch, net, optimizer, MODEL_DIRPATH + f'vqvae-model-epoch{epoch + 1}.pth')