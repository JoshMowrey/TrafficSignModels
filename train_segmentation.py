import os
import json
import glob
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torchvision.transforms.functional as TF
from model import UNet

# Configuration
IMAGE_DIR = 'extracted_data/images'
ANNOTATION_DIR = 'extracted_data/Labels_and_stuff/mtsd_v2_partially_annotated/annotations'
IMG_SIZE = (256, 256)
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_SAMPLES = None  # Set to None to use all data

def calculate_metrics(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    
    # Flatten
    pred = pred.view(-1)
    target = target.view(-1)
    
    # Accuracy
    correct = (pred == target).float().sum()
    accuracy = correct / target.numel()
    
    # IoU
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    if union == 0:
        iou = 1.0 # If both are empty, it's a perfect match
    else:
        iou = intersection / union
        
    return accuracy.item(), iou.item()

# Configuration
IMAGE_DIR = 'extracted_data/images'
ANNOTATION_DIR = 'extracted_data/Labels_and_stuff/mtsd_v2_partially_annotated/annotations'
IMG_SIZE = (512, 512)
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
EPOCHS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_SAMPLES = None  # Set to None to use all data

def calculate_metrics(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    
    # Flatten
    pred = pred.view(-1)
    target = target.view(-1)
    
    # Accuracy
    correct = (pred == target).float().sum()
    accuracy = correct / target.numel()
    
    # IoU
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    if union == 0:
        iou = 1.0 # If both are empty, it's a perfect match
    else:
        iou = intersection / union
        
    return accuracy.item(), iou.item()

class TrafficSignDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None, max_samples=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        
        # We list JSON files as they contain the labels. 
        # We assume every JSON has a corresponding image.
        all_annotation_files = glob.glob(os.path.join(annotation_dir, '*.json'))
        
        if max_samples:
            all_annotation_files = all_annotation_files[:max_samples]
            print(f"Limiting to first {max_samples} annotation files for scanning.")
        
        self.valid_samples = []
        print("Scanning files for US images (suffix '--g1')...")
        
        for ann_path in all_annotation_files:
            # Load JSON to check content
            try:
                with open(ann_path, 'r') as f:
                    data = json.load(f)
                
                # Check for US tag (g1)
                has_us_sign = False
                for obj in data['objects']:
                    if obj['label'].endswith('--g1'):
                        has_us_sign = True
                        break
                
                if has_us_sign:
                    base_name = os.path.splitext(os.path.basename(ann_path))[0]
                    img_path = os.path.join(image_dir, base_name + '.jpg')
                    # We assume image exists to save time, or check:
                    if os.path.exists(img_path):
                        self.valid_samples.append((img_path, ann_path))
                        
            except Exception as e:
                print(f"Error reading {ann_path}: {e}")
                continue
        
        print(f"Found {len(self.valid_samples)} valid US image/annotation pairs.")

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        img_path, ann_path = self.valid_samples[idx]
        
        # Load Image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image and mask as fallback
            return torch.zeros(3, IMG_SIZE[0], IMG_SIZE[1]), torch.zeros(1, IMG_SIZE[0], IMG_SIZE[1])

        w, h = image.size
        
        # Load Annotation and Create Mask
        with open(ann_path, 'r') as f:
            data = json.load(f)
        
        # Create a blank mask (same size as original image)
        mask = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask)
        
        for obj in data['objects']:
            bbox = obj['bbox']
            xmin = bbox['xmin']
            ymin = bbox['ymin']
            xmax = bbox['xmax']
            ymax = bbox['ymax']
            
            # Clamp to image boundaries
            xmin = max(0, min(xmin, w))
            ymin = max(0, min(ymin, h))
            xmax = max(0, min(xmax, w))
            ymax = max(0, min(ymax, h))
            
            # Validate
            if xmax <= xmin or ymax <= ymin:
                continue

            draw.rectangle(
                [xmin, ymin, xmax, ymax], 
                fill=255
            )
        
        # Resize
        image = TF.resize(image, IMG_SIZE)
        mask = TF.resize(mask, IMG_SIZE, interpolation=transforms.InterpolationMode.NEAREST)
        
        # To Tensor
        image = TF.to_tensor(image)
        mask = torch.from_numpy(np.array(mask)).float().unsqueeze(0) / 255.0
        
        return image, mask

def train():
    print(f"Using device: {DEVICE}")
    
    # Prepare Data
    full_dataset = TrafficSignDataset(IMAGE_DIR, ANNOTATION_DIR, max_samples=MAX_SAMPLES)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Model, Loss, Optimizer
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training Loop
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_acc = 0
        train_iou = 0
        
        for images, masks in train_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            acc, iou = calculate_metrics(torch.sigmoid(outputs), masks)
            train_acc += acc
            train_iou += iou
        
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
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                acc, iou = calculate_metrics(torch.sigmoid(outputs), masks)
                val_acc += acc
                val_iou += iou
        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader)
        avg_train_iou = train_iou / len(train_loader)
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}]")
        print(f"  Train Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.4f}, IoU: {avg_train_iou:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, Acc: {avg_val_acc:.4f}, IoU: {avg_val_iou:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'unet_traffic_sign.pth')
            print("  Saved best model.")


if __name__ == '__main__':
    train()
