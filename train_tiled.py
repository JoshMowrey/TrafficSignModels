import glob
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from model import UNet

# Configuration
TRAIN_DIR = "tiled_data/train"
VAL_DIR = "tiled_data/val"
SAVE_MODEL_PATH = "models/model_tiled.pth"

# Training Hyperparameters
IMG_SIZE = (512, 512)
BATCH_SIZE = 8
EFFECTIVE_BATCH_SIZE = 64
ACCUMULATION_STEPS = EFFECTIVE_BATCH_SIZE // BATCH_SIZE
LEARNING_RATE = 1e-4
EPOCHS = 100
PATIENCE = 15  # Increased patience as dataset is cleaner
MIN_DELTA = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Target Categories
TARGET_KEYWORDS = [
    'regulatory--keep-right',
    'complementary--keep-right',
    'warning--traffic-merges',
    'information--pedestrians-crossing',
    'warning--pedestrians-crossing',
    'warning--traffic-signals',
    'regulatory--stop',
    'regulatory--yield',
    'regulatory--maximum-speed-limit'
]

def is_target_label(label):
    for keyword in TARGET_KEYWORDS:
        if keyword in label:
            return True
    return False

def calculate_metrics(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    pred = pred.view(-1)
    target = target.view(-1)
    
    correct = (pred == target).float().sum()
    accuracy = correct / target.numel()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    if union == 0:
        iou = 1.0 
    else:
        iou = intersection / union
        
    return accuracy.item(), iou.item()

class TiledTrafficSignDataset(Dataset):
    def __init__(self, root_dir, augment=False):
        self.image_dir = os.path.join(root_dir, 'images')
        self.annotation_dir = os.path.join(root_dir, 'annotations')
        self.augment = augment
        
        self.annotation_files = glob.glob(os.path.join(self.annotation_dir, '*.json'))
        print(f"Dataset from {root_dir} initialized with {len(self.annotation_files)} samples.")

        # Augmentation transforms (Photometric only here, geometric done manually below)
        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)

    def __len__(self):
        return len(self.annotation_files)

    def __getitem__(self, idx):
        ann_path = self.annotation_files[idx]
        
        try:
            with open(ann_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            return torch.zeros(3, IMG_SIZE[0], IMG_SIZE[1]), torch.zeros(1, IMG_SIZE[0], IMG_SIZE[1])
            
        base_name = os.path.splitext(os.path.basename(ann_path))[0]
        img_path = os.path.join(self.image_dir, base_name + '.jpg')
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            return torch.zeros(3, IMG_SIZE[0], IMG_SIZE[1]), torch.zeros(1, IMG_SIZE[0], IMG_SIZE[1])

        w, h = image.size
        mask = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask)
        
        for obj in data.get('objects', []):
            if is_target_label(obj['label']):
                bbox = obj['bbox']
                xmin = max(0, min(bbox['xmin'], w))
                ymin = max(0, min(bbox['ymin'], h))
                xmax = max(0, min(bbox['xmax'], w))
                ymax = max(0, min(bbox['ymax'], h))
                
                if xmax > xmin and ymax > ymin:
                    draw.rectangle([xmin, ymin, xmax, ymax], fill=255)
        
        # --- Augmentations ---
        if self.augment:
            # 1. Horizontal Flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            
            # 2. Color Jitter (Image only)
            image = self.color_jitter(image)
            
            # 3. Random Rotation
            # (Careful with rotation as it introduces black borders, but +/- 10 deg is usually safe)
            if random.random() > 0.5:
                angle = random.uniform(-10, 10)
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)

        # Resize (Should already be 512, but ensures consistency)
        image = TF.resize(image, IMG_SIZE)
        mask = TF.resize(mask, IMG_SIZE, interpolation=transforms.InterpolationMode.NEAREST)
        
        return TF.to_tensor(image), torch.from_numpy(np.array(mask)).float().unsqueeze(0) / 255.0

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs).view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs).view(-1)
        targets = targets.view(-1)
        bce = - (targets * torch.log(inputs + 1e-8) + (1 - targets) * torch.log(1 - inputs + 1e-8))
        focal_weight = targets * self.alpha * (1 - inputs)**self.gamma + (1 - targets) * (1 - self.alpha) * inputs**self.gamma
        return (focal_weight * bce).mean()

def train():
    print(f"Starting training on device: {DEVICE}")
    print(f"Dataset: Tiled (512x512)")
    
    train_dataset = TiledTrafficSignDataset(TRAIN_DIR, augment=True)
    val_dataset = TiledTrafficSignDataset(VAL_DIR, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    
    focal_criterion = FocalLoss()
    dice_criterion = DiceLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        optimizer.zero_grad()
        
        for i, (images, masks) in enumerate(train_loader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            
            outputs = model(images)
            
            loss = (focal_criterion(outputs, masks) + dice_criterion(outputs, masks)) / ACCUMULATION_STEPS
            loss.backward()
            
            if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss += loss.item() * ACCUMULATION_STEPS
            
            if i % 50 == 0: 
                print(f"  Step [{i}/{len(train_loader)}] Loss: {loss.item() * ACCUMULATION_STEPS:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        val_acc = 0
        val_iou = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)
                outputs = model(images)
                
                loss = focal_criterion(outputs, masks) + dice_criterion(outputs, masks)
                val_loss += loss.item()
                
                acc, iou = calculate_metrics(torch.sigmoid(outputs), masks)
                val_acc += acc
                val_iou += iou

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)

        print(f"Epoch [{epoch+1}/{EPOCHS}] Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, Acc: {avg_val_acc:.4f}, IoU: {avg_val_iou:.4f}")

        if avg_val_loss < (best_val_loss - MIN_DELTA):
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), SAVE_MODEL_PATH)
            print(f"  Validation Loss Improved. Model saved to {SAVE_MODEL_PATH}")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print("  Early stopping.")
                break

if __name__ == "__main__":
    train()
