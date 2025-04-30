import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
# import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from datasets import Datasets
from datasets import load_dataset

class UNET_Generator(nn.Module):
  def __init__(self):
    super(Unet, self).__init__()
    self.layer1 = nn.Sequential(
        nn.Conv2d(3, 64, 3),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3),
        nn.ReLU()
    )
    self.layer2 = nn.Sequential(
        nn.MaxPool2d(2,2),
        nn.Conv2d(64, 128, 3),
        nn.ReLU(),
        nn.Conv2d(128, 128, 3),
        nn.ReLU()
    )
    self.layer3 = nn.Sequential(
        nn.MaxPool2d(2,2),
        nn.Conv2d(128, 256, 3),
        nn.ReLU(),
        nn.Conv2d(256, 256, 3),
        nn.ReLU()
    )
    self.layer4 = nn.Sequential(
        nn.MaxPool2d(2,2),
        nn.Conv2d(256, 512, 3),
        nn.ReLU(),
        nn.Conv2d(512, 512, 3),
        nn.ReLU()
    )
    self.layer5 = nn.Sequential(
        nn.MaxPool2d(2,2),
        nn.Conv2d(512, 1024, 3),
        nn.ReLU(),
        nn.Conv2d(1024, 1024, 3),
        nn.ReLU()
    )
    # apply up conv and copy
    self.upconv1 = nn.ConvTranspose2d(1024, 512, 2,2)
    self.layer6 = nn.Sequential(
        nn.Conv2d(1024, 512, 3),
        nn.ReLU(),
        nn.Conv2d(512, 512, 3),
        nn.ReLU()
    )
    # apply up conv and copy
    self.upconv2 = nn.ConvTranspose2d(512, 256, 2,2)
    self.layer7 = nn.Sequential(
        nn.Conv2d(512, 256, 3),
        nn.ReLU(),
        nn.Conv2d(256, 256, 3),
        nn.ReLU()
    )
    # apply up conv and copy
    self.upconv3 = nn.ConvTranspose2d(256, 128, 2,2)
    self.layer8 = nn.Sequential(
        nn.Conv2d(256, 128, 3),
        nn.ReLU(),
        nn.Conv2d(128, 128, 3),
        nn.ReLU()
    )
    # apply up conv and copy
    self.upconv4 = nn.ConvTranspose2d(128, 64, 2,2)
    self.layer9 = nn.Sequential(
        nn.Conv2d(128, 64, 3),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3),
        nn.ReLU(),
        nn.Conv2d(64, 1, 1)
    )

  def forward(self, x):
    x1 = self.layer1(x)
    x2 = self.layer2(x1)
    x3 = self.layer3(x2)
    x4 = self.layer4(x3)
    x = self.layer5(x4)
    x = self.upconv1(x)
    x4 = TF.resize(x4, size=x.shape[2:])
    x = torch.cat((x4, x), dim=1)
    x = self.layer6(x)
    x = self.upconv2(x)
    x3 = TF.resize(x3, size=x.shape[2:])
    x = torch.cat((x3,x), dim=1)
    x = self.layer7(x)
    x = self.upconv3(x)
    x2 = TF.resize(x2, size=x.shape[2:])
    x = torch.cat((x2, x), dim=1)
    x = self.layer8(x)
    x = self.upconv4(x)
    x1 = TF.resize(x1, size=x.shape[2:])
    x = torch.cat((x1,x), dim=1)
    x = self.layer9(x)
    return x
  
class PatchDiscirminator(nn.Module):
  def __init__(self):
    super(PatchDiscriminator, self).__init__()
    self.conv1 = nn.Conv2d(3, 8, 5)
    self.pool = nn.MaxPool2d(2,2)
    self.conv2 = nn.Conv2d(8, 16, 5)
    self.fc1 = nn.Lienar(5*5*16, 128)
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, 16)
    self.fc4 = nn.Linear(16,2)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = self.pool(x)
    x = F.relu(self.conv2(x))
    x = self.pool(x)
    x = x.view(-1, 5*5*16)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = F.sigmoid(self.fc4(x))
    return x
  
class Discriminator(nn.Module):
  def __init__(self, N):
    super(Discriminator, self).__init__()
    self.PatchD = PatchDiscriminator()
    self.N = N

  def forward(self, x):
    patch = x.shape[0]//self.N
    avg = 0
    for i in range(x.shape[0], patch):
      for j in tange(x.shape[1], patch):
        avg += (self.PatchD(x[i:i+patch, j:j+patch, :]))/(self.N*self.N)
    return avg