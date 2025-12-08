import numpy as np
import random
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from torchvision import transforms

# Import architectures
from model import UNet as StandardUNet
try:
    from model_efficientnet import EfficientNetUNetFixed
except ImportError:
    EfficientNetUNetFixed = None

# Import Dataset (Reuse from train_filtered.py logic to avoid duplication, 
# but cleaner to redefine minimal dataset class to be self-contained)
from torch.utils.data import Dataset
import glob
import json
from PIL import Image, ImageDraw

# --- CONFIGURATION ---
TRAIN_DIR = "tiled_data/train"
VAL_DIR = "tiled_data/val"
IMG_SIZE = (512, 512) # Tiled data is 512x512 native
BATCH_SIZE = 8        # Increased batch size for 512 tiles
EFFECTIVE_BATCH_SIZE = 64
ACCUMULATION_STEPS = EFFECTIVE_BATCH_SIZE // BATCH_SIZE
FINE_TUNE_LR = 1e-5 
EPOCHS = 20         
PATIENCE = 5
MIN_DELTA = 0.0001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TARGET_KEYWORDS = [
    'regulatory--stop', 'regulatory--yield', 'regulatory--keep-right', 
    'complementary--keep-right', 'warning--traffic-merges', 
    'information--pedestrians-crossing', 'warning--pedestrians-crossing', 
    'warning--traffic-signals', 'regulatory--maximum-speed-limit'
]

# --- HELPERS ---
def is_target_label(label):
    for keyword in TARGET_KEYWORDS:
        if keyword in label: return True
    return False

def calculate_metrics(pred, target, threshold=0.5):
    pred = (pred > threshold).float().view(-1)
    target = target.view(-1)
    correct = (pred == target).float().sum()
    acc = correct / target.numel()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    iou = 1.0 if union == 0 else inter / union
    return acc.item(), iou.item()

class FilteredTrafficSignDataset(Dataset):
    def __init__(self, root_dir, augment=False):
        self.image_dir = os.path.join(root_dir, 'images')
        self.annotation_dir = os.path.join(root_dir, 'annotations')
        self.annotation_files = glob.glob(os.path.join(self.annotation_dir, '*.json'))
        self.augment = augment
        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)

    def __len__(self): return len(self.annotation_files)

    def __getitem__(self, idx):
        ann_path = self.annotation_files[idx]
        try:
            with open(ann_path, 'r') as f: data = json.load(f)
        except: return torch.zeros(3, IMG_SIZE[0], IMG_SIZE[1]), torch.zeros(1, IMG_SIZE[0], IMG_SIZE[1])
        
        base = os.path.splitext(os.path.basename(ann_path))[0]
        img_path = os.path.join(self.image_dir, base + '.jpg')
        try: image = Image.open(img_path).convert("RGB")
        except: return torch.zeros(3, IMG_SIZE[0], IMG_SIZE[1]), torch.zeros(1, IMG_SIZE[0], IMG_SIZE[1])

        w, h = image.size
        mask = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask)
        for obj in data.get('objects', []):
            if is_target_label(obj['label']):
                b = obj['bbox']
                draw.rectangle([max(0, min(b['xmin'], w)), max(0, min(b['ymin'], h)), 
                                max(0, min(b['xmax'], w)), max(0, min(b['ymax'], h))], fill=255)
        
        # Augmentations
        if self.augment:
            import random
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            image = self.color_jitter(image)
            if random.random() > 0.5:
                angle = random.uniform(-10, 10)
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)

        image = TF.resize(image, IMG_SIZE)
        mask = TF.resize(mask, IMG_SIZE, interpolation=transforms.InterpolationMode.NEAREST)
        return TF.to_tensor(image), torch.from_numpy(np.array(mask)).float().unsqueeze(0) / 255.0

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super().__init__()
        self.smooth = smooth
    def forward(self, i, t):
        i, t = torch.sigmoid(i).view(-1), t.view(-1)
        inter = (i * t).sum()
        return 1 - (2. * inter + self.smooth) / (i.sum() + t.sum() + self.smooth)

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

def load_model_safely(path):
    print(f"Attempting to load model from {path}...")
    state_dict = torch.load(path, map_location=DEVICE)
    
    # 1. Try EfficientNet
    if EfficientNetUNetFixed:
        try:
            model = EfficientNetUNetFixed(in_channels=3, out_channels=1).to(DEVICE)
            model.load_state_dict(state_dict)
            print("  -> Detected Architecture: EfficientNet-B0 U-Net")
            return model
        except Exception:
            pass 

    # 2. Try Standard UNet
    try:
        model = StandardUNet(in_channels=3, out_channels=1).to(DEVICE)
        model.load_state_dict(state_dict)
        print("  -> Detected Architecture: Standard UNet")
        return model
    except Exception:
        pass

    print("Error: Could not match model architecture to state dict.")
    return None

def train(model_path, output_name=None):
    # Setup Paths
    if output_name:
        if not output_name.endswith('.pth'):
            output_name += '.pth'
        save_path = os.path.join("models", output_name)
    else:
        base_name = os.path.splitext(os.path.basename(model_path))[0]
        save_path = f"models/{base_name}_finetuned.pth"
    
    # Load Data
    print("Loading Datasets (Tiled)...")
    train_ds = FilteredTrafficSignDataset(TRAIN_DIR, augment=True)
    val_ds = FilteredTrafficSignDataset(VAL_DIR, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Load Model
    model = load_model_safely(model_path)
    if not model: return

    # Setup Training
    # pos_weight = torch.tensor([20.0]).to(DEVICE) # Removed for Focal Loss
    # bce_crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    focal_crit = FocalLoss(alpha=0.8, gamma=2.0) # Alpha=0.8 biases towards positive class (signs)
    dice_crit = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=FINE_TUNE_LR) # Low LR
    
    best_val_loss = float('inf')
    patience = 0
    
    print(f"Starting Fine-Tuning: {EPOCHS} Epochs, LR={FINE_TUNE_LR}")
    print(f"Image Size: {IMG_SIZE}, Batch: {BATCH_SIZE}, Accum: {ACCUMULATION_STEPS}")
    print(f"Saving to: {save_path}")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        
        for i, (imgs, masks) in enumerate(train_loader):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            out = model(imgs)
            loss = (focal_crit(out, masks) + dice_crit(out, masks)) / ACCUMULATION_STEPS
            loss.backward()
            
            if (i+1) % ACCUMULATION_STEPS == 0 or (i+1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss += loss.item() * ACCUMULATION_STEPS
            
            if i % 50 == 0:
                print(f"  Step {i}/{len(train_loader)} Loss: {loss.item()*ACCUMULATION_STEPS:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        val_acc = 0
        val_iou = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                out = model(imgs)
                loss = focal_crit(out, masks) + dice_crit(out, masks)
                val_loss += loss.item()
                a, iou = calculate_metrics(torch.sigmoid(out), masks)
                val_acc += a
                val_iou += iou
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        avg_acc = val_acc / len(val_loader)
        avg_iou = val_iou / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss {avg_train:.4f} | Val Loss {avg_val:.4f} | Acc {avg_acc:.4f} | IoU {avg_iou:.4f}")
        
        if avg_val < (best_val_loss - MIN_DELTA):
            best_val_loss = avg_val
            patience = 0
            torch.save(model.state_dict(), save_path)
            print("  -> Improved! Saved.")
        else:
            patience += 1
            print(f"  -> No improvement. Patience {patience}/{PATIENCE}")
            if patience >= PATIENCE:
                print("Early Stopping.")
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune a traffic sign segmentation model")
    parser.add_argument("model_path", type=str, help="Path to the .pth model file")
    parser.add_argument("--output_name", type=str, default=None, help="Custom name for the saved model")
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: File {args.model_path} not found.")
    else:
        train(args.model_path, args.output_name)
