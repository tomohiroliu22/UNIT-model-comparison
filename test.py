import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--dataroot", type=str, default="your own path", help="your own path to dataset")
parser.add_argument("--epoch", type=int, default=299, help="epoch to start training from")
parser.add_argument("--in_channel", type=int, default=3, help="input data dimension 3:RGB / 1:GRAY")
parser.add_argument("--out_channel", type=int, default=1, help="output data dimension 3:RGB / 1:GRAY")
parser.add_argument("--dataset_name", type=str, default="apple2orange", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between saving model checkpoints")
parser.add_argument("--n_downsample", type=int, default=2, help="number downsampling layers in encoder")
parser.add_argument("--dim", type=int, default=64, help="number of filters in first encoder layer")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False

# Create sample and checkpoint directories
os.makedirs("testing/%s" % opt.dataset_name, exist_ok=True)

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_pixel = torch.nn.L1Loss()

input_shape = (opt.channels, opt.img_height, opt.img_width)

# Dimensionality (channel-wise) of image embedding
shared_dim = opt.dim * 2 ** opt.n_downsample

# Initialize generator and discriminator
shared_E = ResidualBlock(features=shared_dim)
E1 = Encoder(dim=opt.dim, in_channels=opt.in_channel, n_downsample=opt.n_downsample, shared_block=shared_E)
E2 = Encoder(dim=opt.dim, in_channels=opt.out_channel, n_downsample=opt.n_downsample, shared_block=shared_E)
shared_G = ResidualBlock(features=shared_dim)
G1 = Generator(dim=opt.dim, out_channels=opt.in_channel, n_upsample=opt.n_downsample, shared_block=shared_G)
G2 = Generator(dim=opt.dim, out_channels=opt.out_channel, n_upsample=opt.n_downsample, shared_block=shared_G)
D1 = Discriminator(channels = opt.in_channel)
D2 = Discriminator(channels = opt.out_channel)

if cuda:
    E1 = E1.cuda()
    E2 = E2.cuda()
    G1 = G1.cuda()
    G2 = G2.cuda()
    D1 = D1.cuda()
    D2 = D2.cuda()
    criterion_GAN.cuda()
    criterion_pixel.cuda()

# Load pretrained models
E1.load_state_dict(torch.load("saved_models/%s/E1_%d.pth" % (opt.dataset_name, opt.epoch)))
E2.load_state_dict(torch.load("saved_models/%s/E2_%d.pth" % (opt.dataset_name, opt.epoch)))
G1.load_state_dict(torch.load("saved_models/%s/G1_%d.pth" % (opt.dataset_name, opt.epoch)))
G2.load_state_dict(torch.load("saved_models/%s/G2_%d.pth" % (opt.dataset_name, opt.epoch)))
D1.load_state_dict(torch.load("saved_models/%s/D1_%d.pth" % (opt.dataset_name, opt.epoch)))
D2.load_state_dict(torch.load("saved_models/%s/D2_%d.pth" % (opt.dataset_name, opt.epoch)))

# Loss weights
lambda_0 = 10  # GAN
lambda_1 = 0.1  # KL (encoded images)
lambda_2 = 100  # ID pixel-wise
lambda_3 = 0.1  # KL (encoded translated images)
lambda_4 = 100  # Cycle pixel-wise

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Dataset loader
transforms_ = [
    transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
dataloader = DataLoader(
    ImageDataset(opt.dataroot+"/%s" % opt.dataset_name, transforms_=transforms_, unaligned=True),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)
val_dataloader = DataLoader(
    ImageDataset(opt.dataroot+"/%s" % opt.dataset_name, transforms_=transforms_, unaligned=True, mode="test"),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

# ----------
#  Testing
# ----------
batches_done = 0
prev_time = time.time()
for i, batch in enumerate(val_dataloader):
    X1 = Variable(batch["A"].type(Tensor))
    X2 = Variable(batch["B"].type(Tensor))
    E1.eval()
    E2.eval()
    G1.eval()
    G2.eval()
    _, Z1 = E1(X1)
    _, Z2 = E2(X2)
    fake_X1 = G1(Z2)
    fake_X2 = G2(Z1)
    save_image(X1, "testing/%s/%s_real_A.png" % (opt.dataset_name, batches_done), nrow=1, normalize=True)
    save_image(X2, "testing/%s/%s_real_B.png" % (opt.dataset_name, batches_done), nrow=1, normalize=True)
    save_image(fake_X1, "testing/%s/%s_fake_A.png" % (opt.dataset_name, batches_done), nrow=1, normalize=True)
    save_image(fake_X2, "testing/%s/%s_fake_B.png" % (opt.dataset_name, batches_done), nrow=1, normalize=True)
    batches_done+=1
