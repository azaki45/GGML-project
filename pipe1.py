import os
import numpy as np
from numpy import load, savez_compressed, ones, zeros
from numpy.random import randint
from matplotlib import pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


##############################################
# Data loading and preprocessing functions
##############################################

# This function emulates the load_images in your Keras code.
def load_images(path, size=(256, 512)):
    src_imgs, targ_imgs = list(), list()
    for filename in os.listdir(path):
        img_path = os.path.join(path, filename)
        image = Image.open(img_path).resize(size)  # size: (width, height)
        pixels = np.array(image)
        # Split the image in half (assumes width is split equally)
        edge_image = pixels[:, :256, :].astype('float32')
        real_image = pixels[:, 256:, :].astype('float32')
        src_imgs.append(edge_image)
        targ_imgs.append(real_image)
    return np.asarray(src_imgs), np.asarray(targ_imgs)

# Uncomment and use if you want to re-create the dataset:
# path = "/facades/train"
# print(path)
# src_imageset, targ_imageset = load_images(path, size=(256,512))
# print("loaded:", src_imageset.shape, targ_imageset.shape)
# filename = "facades.npz"
# savez_compressed(filename, src_imageset, targ_imageset)
# print("saved dataset", filename)

# Load and preprocess dataset from NPZ
def load_dataset(filename):
    data = load(filename)
    X1, X2 = data["arr_0"], data["arr_1"]
    # Normalize to [-1, 1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    # Convert images from (N, H, W, C) to (N, C, H, W)
    X1 = np.transpose(X1, (0, 3, 1, 2))
    X2 = np.transpose(X2, (0, 3, 1, 2))
    return X1, X2


##############################################
# Weight initialization: mimic RandomNormal(stddev=0.02)
##############################################
def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


##############################################
# Discriminator definition
##############################################

class Discriminator(nn.Module):
    def __init__(self, image_shape):
        """
        image_shape: tuple (channels, height, width), e.g., (3, 256, 256)
        """
        super(Discriminator, self).__init__()
        # In Keras, the discriminator concatenates source and target images along channels.
        # Thus, input channels = 3 + 3 = 6.
        in_channels = image_shape[0] * 2
        
        self.model = nn.Sequential(
            # Layer 1: Conv2d with stride 2
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 2:
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 3:
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 4:
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 5: no stride (assumed stride=1), padding chosen to mimic 'same'
            nn.Conv2d(512, 512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 6: final convolution, output 1 channel
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, src, target):
        # Concatenate along channel axis
        x = torch.cat([src, target], dim=1)
        return self.model(x)


##############################################
# Generator definition (U-Net based)
##############################################

# Encoder block: Conv2d with stride 2, optional BatchNorm then LeakyReLU
def encoder_block(in_channels, out_channels, kernel_size=4, batchnorm=True):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=1)]
    if batchnorm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)

# Decoder block: ConvTranspose2d followed by BatchNorm, optional Dropout, concatenation with skip connection and ReLU.
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=True):
        super(DecoderBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.use_dropout = dropout
        if dropout:
            self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, skip):
        x = self.deconv(x)
        x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        # Concatenate with skip connection along channel dimension
        x = torch.cat([x, skip], dim=1)
        x = self.relu(x)
        return x

class Generator(nn.Module):
    def __init__(self, input_img_shape, batchnorm=True, dropout=True):
        """
        input_img_shape: tuple (channels, H, W) e.g., (3, 256, 256)
        """
        super(Generator, self).__init__()
        channels = input_img_shape[0]
        # Contracting Path (Encoder)
        self.enc1 = encoder_block(channels, 64, kernel_size=4, batchnorm=False)  # c1
        self.enc2 = encoder_block(64, 128, kernel_size=4, batchnorm=batchnorm)   # c2
        self.enc3 = encoder_block(128, 256, kernel_size=4, batchnorm=batchnorm)  # c3
        self.enc4 = encoder_block(256, 512, kernel_size=4, batchnorm=batchnorm)  # c4
        self.enc5 = encoder_block(512, 512, kernel_size=4, batchnorm=batchnorm)  # c5
        self.enc6 = encoder_block(512, 512, kernel_size=4, batchnorm=batchnorm)  # c6
        self.enc7 = encoder_block(512, 512, kernel_size=4, batchnorm=batchnorm)  # c7
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Expansive Path (Decoder)
        self.dec1 = DecoderBlock(512, 512, dropout=dropout)  # d1, input from bottleneck, skip: c7
        self.dec2 = DecoderBlock(1024, 512, dropout=dropout)   # d2, input channels doubled due to concat
        self.dec3 = DecoderBlock(1024, 512, dropout=dropout)   # d3, skip: from c5
        self.dec4 = DecoderBlock(1024, 512, dropout=False)       # d4, skip: from c4
        self.dec5 = DecoderBlock(1024, 256, dropout=False)       # d5, skip: from c3
        self.dec6 = DecoderBlock(512, 128, dropout=False)        # d6, skip: from c2
        self.dec7 = DecoderBlock(256, 64, dropout=False)         # d7, skip: from c1
        
        # Final output layer: note that after dec7, the number of channels is (output of deconv) + (channels from skip).
        # However, in the original architecture, the final layer only applies a ConvTranspose2d to dec7 output.
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # Encoder
        c1 = self.enc1(x)  # (64)
        c2 = self.enc2(c1) # (128)
        c3 = self.enc3(c2) # (256)
        c4 = self.enc4(c3) # (512)
        c5 = self.enc5(c4) # (512)
        c6 = self.enc6(c5) # (512)
        c7 = self.enc7(c6) # (512)
        
        # Bottleneck
        m = self.bottleneck(c7)  # (512)
        
        # Decoder: note that the concatenation doubles the channels for the next block
        d1 = self.dec1(m, c7)       # d1: output channels = 512 + 512 = 1024 after cat, then ReLU applied inside block
        d2 = self.dec2(d1, c6)      # d2: input channels = 1024+512 = 1536 -> but block expects input channels set by design.
        d3 = self.dec3(d2, c5)
        d4 = self.dec4(d3, c4)
        d5 = self.dec5(d4, c3)
        d6 = self.dec6(d5, c2)
        d7 = self.dec7(d6, c1)
        
        out = self.final(d7)
        return out


##############################################
# Helper functions for generating real & fake images
##############################################
def generate_real_images(dataset, n, patch_shape):
    trainA, trainB = dataset
    ix = randint(0, trainA.shape[0], n)
    X1, X2 = trainA[ix], trainB[ix]
    # Create a tensor for "real" labels; patch_shape is the height/width of discriminator output
    y = np.ones((n, patch_shape, patch_shape, 1), dtype=np.float32)
    # Convert images from numpy to torch tensors
    X1 = torch.from_numpy(X1).float().to(device)
    X2 = torch.from_numpy(X2).float().to(device)
    y = torch.from_numpy(y).float().to(device)
    return (X1, X2), y

def generate_fake_images(model, sample, patch_shape):
    # sample is a batch of source images (torch tensor)
    with torch.no_grad():
        X = model(sample)
    n = X.shape[0]
    y = torch.zeros((n, patch_shape, patch_shape, 1), dtype=torch.float32, device=device)
    return X, y


##############################################
# Performance check: Plot and save images & model
##############################################
def performance_check(step, gen_model, dataset):
    (realA, realB), _ = generate_real_images(dataset, 3, patch_shape=1)
    fakeB, _ = generate_fake_images(gen_model, realA, patch_shape=1)
    # Denormalize images from [-1,1] to [0,1]
    realA = (realA.cpu().numpy() + 1) / 2.0
    realB = (realB.cpu().numpy() + 1) / 2.0
    fakeB = (fakeB.cpu().numpy() + 1) / 2.0
    
    n = 3
    plt.figure(figsize=(12, 8))
    # Plot real source images
    for i in range(n):
        plt.subplot(3, n, 1+i)
        plt.axis("off")
        # Transpose back to (H,W,C)
        plt.imshow(np.transpose(realA[i], (1,2,0)))
    # Plot generated target images
    for i in range(n):
        plt.subplot(3, n, 1+n+i)
        plt.axis("off")
        plt.imshow(np.transpose(fakeB[i], (1,2,0)))
    # Plot real target images
    for i in range(n):
        plt.subplot(3, n, 1+n*2+i)
        plt.axis("off")
        plt.imshow(np.transpose(realB[i], (1,2,0)))
        
    filename1 = 'plot_{:06d}.png'.format(step+1)
    plt.savefig(filename1)
    plt.close()
    # Save generator model state
    filename2 = 'model_{:06d}.pth'.format(step+1)
    torch.save(gen_model.state_dict(), filename2)
    print(">saved: {} and {}".format(filename1, filename2))
    

##############################################
# Training loop
##############################################
def train_gan(disc_model, gen_model, dataset, epochs=100, batch=1):
    # Get dataset arrays
    trainA, trainB = dataset
    batch_per_epoch = int(trainA.shape[0] / batch)
    steps = batch_per_epoch * epochs
    print("Batches per epoch:", batch_per_epoch)
    print("Total steps:", steps)
    
    # Assume the output of the discriminator has shape (N, 1, patch_shape, patch_shape).
    # We determine patch_shape by doing a forward pass on one sample.
    disc_model.eval()
    with torch.no_grad():
        sample_src = torch.from_numpy(trainA[0:1]).float().to(device)
        sample_targ = torch.from_numpy(trainB[0:1]).float().to(device)
        out = disc_model(sample_src, sample_targ)
    patch_shape = out.shape[2]  # assuming square output
    disc_model.train()
    
    # Loss functions
    criterion_GAN = nn.BCELoss()
    criterion_L1 = nn.L1Loss()
    
    # Optimizers
    optimizer_D = optim.Adam(disc_model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_G = optim.Adam(gen_model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    for i in range(steps):
        # ---------------------
        #  Train Discriminator
        # ---------------------
        # Sample a batch of real images
        (realA, realB), y_real = generate_real_images(dataset, batch, patch_shape)
        
        # Generate fake images
        fakeB = gen_model(realA)
        
        # Train on real images
        optimizer_D.zero_grad()
        pred_real = disc_model(realA, realB)
        loss_real = criterion_GAN(pred_real, y_real)
        
        # Train on fake images (detach generator)
        pred_fake = disc_model(realA, fakeB.detach())
        y_fake = torch.zeros_like(y_real, device=device)
        loss_fake = criterion_GAN(pred_fake, y_fake)
        
        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizer_D.step()
        
        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()
        # Generator wants discriminator to output ones on fake images
        pred_fake_for_G = disc_model(realA, fakeB)
        loss_G_GAN = criterion_GAN(pred_fake_for_G, y_real)
        loss_G_L1 = criterion_L1(fakeB, realB)
        # Total generator loss with loss weight for L1 = 100
        loss_G = loss_G_GAN + 100 * loss_G_L1
        loss_G.backward()
        optimizer_G.step()
        
        # Print progress
        print(">%d, d_loss_real: %.3f d_loss_fake: %.3f, g_loss: %.3f" % (i+1, loss_real.item(), loss_fake.item(), loss_G.item()))
        
        # Performance check every 10 epochs (or adjust as desired)
        if (i+1) % (batch_per_epoch * 10) == 0:
            performance_check(i, gen_model, dataset)


##############################################
# Main execution
##############################################
if __name__ == '__main__':
    # Load dataset from facades.npz
    dataset = load_dataset('facades.npz')
    print("Loaded dataset shapes:", dataset[0].shape, dataset[1].shape)
    
    # Determine image shape (channels, height, width)
    image_shape = dataset[0].shape[1:]
    
    # Create models
    disc_model = Discriminator(image_shape).to(device)
    gen_model = Generator(image_shape, batchnorm=True, dropout=True).to(device)
    
    # Apply weight initialization to both models
    disc_model.apply(weights_init)
    gen_model.apply(weights_init)
    
    # Train GAN
    train_gan(disc_model, gen_model, dataset, epochs=100, batch=1)
