import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
import os
import random
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt


# ---- VGG16-based Encoder ----
class VGGEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        vgg16 = models.vgg16_bn(pretrained=pretrained).features
        self.enc1 = vgg16[:6]   # 3 -> 64
        self.enc2 = vgg16[6:13]  # 64 -> 128
        self.enc3 = vgg16[13:23]  # 128 -> 256
        self.enc4 = vgg16[23:33]  # 256 -> 512
        self.enc5 = vgg16[33:43]  # 512 -> 512
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, x):
        indices_list = []
        x = self.enc1(x); x, i1 = self.pool(x); indices_list.append(i1)
        x = self.enc2(x); x, i2 = self.pool(x); indices_list.append(i2)
        x = self.enc3(x); x, i3 = self.pool(x); indices_list.append(i3)
        x = self.enc4(x); x, i4 = self.pool(x); indices_list.append(i4)
        x = self.enc5(x); x, i5 = self.pool(x); indices_list.append(i5)
        return x, indices_list


# ---- Decoder with Unpooling ----
class UNetDecoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.dec5 = self.conv_block(channels[0], channels[1])
        self.dec4 = self.conv_block(channels[1], channels[2])
        self.dec3 = self.conv_block(channels[2], channels[3])
        self.dec2 = self.conv_block(channels[3], channels[4])
        self.dec1 = self.conv_block(channels[4], channels[5])

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, indices_list):
        x = self.unpool(x, indices_list[4]); x = self.dec5(x)
        x = self.unpool(x, indices_list[3]); x = self.dec4(x)
        x = self.unpool(x, indices_list[2]); x = self.dec3(x)
        x = self.unpool(x, indices_list[1]); x = self.dec2(x)
        x = self.unpool(x, indices_list[0]); x = self.dec1(x)
        return x


# ---- SegUNet ----
class SegUNet(nn.Module):
    def __init__(self, num_classes=2, dropout=0.05):
        super().__init__()
        self.encoder = VGGEncoder(pretrained=True)
        self.decoder = UNetDecoder([512, 512, 256, 128, 64, 64])
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x, indices_list = self.encoder(x)
        x = self.decoder(x, indices_list)
        x = self.dropout(x)
        x = self.final_conv(x)
        return x


# ---- DataLoader ----
class LungTumorDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].long()

        return image, mask


# ---- Training & Evaluation ----
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=60, patience=10):
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for images, masks in tqdm(train_loader):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss = validate_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "best_segunet.pth")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break


def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

    return val_loss / len(val_loader)


# ---- Main Execution ----
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data Paths
    train_images = ["path/to/train/image1.png", "path/to/train/image2.png"]
    train_masks = ["path/to/train/mask1.png", "path/to/train/mask2.png"]

    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(),
        ToTensorV2()
    ])

    train_dataset = LungTumorDataset(train_images, train_masks, transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Model Setup
    model = SegUNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1.0)

    # Train Model
    train_model(model, train_loader, train_loader, criterion, optimizer, device)

