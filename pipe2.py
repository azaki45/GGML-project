"""
FastCUT implementation for Cityscapes dataset
Based on: https://github.com/taesungp/contrastive-unpaired-translation
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim import Adam
import os
from PIL import Image

# --------------------------
# Dataset Configuration
# --------------------------

class CityscapesDataset(Dataset):
    def __init__(self, root_dir, phase='train', transform=None):
        self.root = os.path.join(root_dir, phase)
        self.img_dir = os.path.join(self.root, 'images')
        self.labels_dir = os.path.join(self.root, 'labels')
        self.transform = transform or self.get_default_transform()
        
        self.files = [f for f in os.listdir(self.img_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.files[idx])
        label_path = os.path.join(self.labels_dir, self.files[idx])
        
        image = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('RGB')
        
        return {'A': self.transform(label), 'B': self.transform(image)}

    @staticmethod
    def get_default_transform():
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

# --------------------------
# Model Architecture
# --------------------------

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels)
        )

    def forward(self, x):
        return x + self.block(x)

class FastCUTGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, n_blocks=9):
        super().__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]
        
        # Downsampling
        in_channels = 64
        out_channels = in_channels*2
        for _ in range(2):
            model += [
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
            in_channels = out_channels
            out_channels = in_channels*2

        # Residual blocks
        for _ in range(n_blocks):
            model += [ResidualBlock(in_channels)]

        # Upsampling
        out_channels = in_channels//2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
            in_channels = out_channels
            out_channels = in_channels//2

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, 7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# --------------------------
# Training Pipeline
# --------------------------

class FastCUTTrainer:
    def __init__(self, args):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize networks
        self.G = FastCUTGenerator().to(self.device)
        self.D = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, padding=1)
        ).to(self.device)
        
        # Optimizers
        self.opt_G = Adam(self.G.parameters(), lr=args.lr, betas=(0.5, 0.999))
        self.opt_D = Adam(self.D.parameters(), lr=args.lr, betas=(0.5, 0.999))
        
        # Losses
        self.criterion_gan = nn.MSELoss()
        self.criterion_nce = PatchNCELoss(args.nce_layers, args.num_patches, args.tau).to(self.device)
        
        # Data loading
        self.dataloader = DataLoader(
            CityscapesDataset(args.data_root),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4
        )

    def train(self, epochs):
        for epoch in range(epochs):
            for batch in self.dataloader:
                real_A = batch['A'].to(self.device)
                real_B = batch['B'].to(self.device)
                
                # Update generators
                self.opt_G.zero_grad()
                fake_B = self.G(real_A)
                loss_G = self.compute_generator_loss(real_A, fake_B)
                loss_G.backward()
                self.opt_G.step()
                
                # Update discriminators
                self.opt_D.zero_grad()
                loss_D = self.compute_discriminator_loss(real_B, fake_B)
                loss_D.backward()
                self.opt_D.step()

            print(f"Epoch {epoch+1}/{epochs} | G Loss: {loss_G.item():.4f} | D Loss: {loss_D.item():.4f}")

    def compute_generator_loss(self, real_A, fake_B):
        # Adversarial loss
        pred_fake = self.D(fake_B)
        loss_gan = self.criterion_gan(pred_fake, torch.ones_like(pred_fake))
        
        # Contrastive loss
        feat_q = self.G.encoder(real_A)
        feat_k = self.G.encoder(fake_B)
        loss_nce = self.criterion_nce(feat_q, feat_k)
        
        return loss_gan + 5.0 * loss_nce

    def compute_discriminator_loss(self, real, fake):
        pred_real = self.D(real)
        loss_real = self.criterion_gan(pred_real, torch.ones_like(pred_real))
        
        pred_fake = self.D(fake.detach())
        loss_fake = self.criterion_gan(pred_fake, torch.zeros_like(pred_fake))
        
        return (loss_real + loss_fake) * 0.5

# --------------------------
# Loss Functions
# --------------------------

class PatchNCELoss(nn.Module):
    def __init__(self, nce_layers=5, num_patches=256, tau=0.07):
        super().__init__()
        self.nce_layers = list(range(nce_layers))
        self.tau = tau
        self.num_patches = num_patches
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, feat_q, feat_k):
        total_loss = 0.0
        for f_q, f_k in zip(feat_q, feat_k):
            B, C, H, W = f_q.shape
            f_q = f_q.view(B, C, -1)
            f_k = f_k.view(B, C, -1)
            
            # Positive logits
            l_pos = torch.bmm(f_q.transpose(1, 2), f_k)
            
            # Negative logits
            l_neg = torch.bmm(f_q.transpose(1, 2), f_k.transpose(1, 2))
            
            # Combine and compute loss
            logits = torch.cat([l_pos, l_neg], dim=2) / self.tau
            labels = torch.zeros(B*H*W, dtype=torch.long).to(f_q.device)
            
            loss = self.cross_entropy(logits.view(-1, logits.size(-1)), labels)
            total_loss += loss.mean()
            
        return total_loss / len(self.nce_layers)

# --------------------------
# Main Execution
# --------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='Path to Cityscapes dataset')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--nce_layers', type=int, default=5)
    parser.add_argument('--num_patches', type=int, default=256)
    parser.add_argument('--tau', type=float, default=0.07)
    args = parser.parse_args()

    trainer = FastCUTTrainer(args)
    trainer.train(args.epochs)
