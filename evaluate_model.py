import os
import json
import glob
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torchvision.transforms.functional as TF
from model import UNet

# Configuration
TEST_DIR = 'filtered_data/test'
MODEL_PATH = 'models/model_filtered.pth' # Use the newly trained model
IMG_SIZE = (512, 512) # Consistent with training
BATCH_SIZE = 8 # Consistent with training
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Target Categories (Must match the filter and training script logic)
TARGET_KEYWORDS = [
    'regulatory--stop',
    'regulatory--yield',
    'regulatory--keep-right',
    'complementary--keep-right',
    'warning--traffic-merges',
    'information--pedestrians-crossing',
    'warning--pedestrians-crossing',
    'warning--traffic-signals',
    'regulatory--maximum-speed-limit'
]

def is_target_label(label):
    """Checks if a label matches any of the target keywords."""
    for keyword in TARGET_KEYWORDS:
        if keyword in label:
            return True
    return False

def calculate_metrics(pred, target, threshold=0.5):
    # Ensure pred and target are on CPU before converting to numpy
    pred = pred.cpu()
    target = target.cpu()

    # Apply sigmoid to predictions and threshold to get binary masks
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    
    # Flatten
    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)
    
    # Accuracy
    correct = (pred_flat == target_flat).float().sum()
    accuracy = correct / target_flat.numel()
    
    # IoU
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    
    if union == 0:
        # If both pred and target are empty, it's a perfect match (IoU = 1.0)
        # If one is empty and other is not, it's 0.0, handled below.
        iou = 1.0 if intersection == 0 else 0.0
    else:
        iou = intersection / union
        
    return accuracy.item(), iou.item()


class FilteredTrafficSignDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_dir = os.path.join(root_dir, 'images')
        self.annotation_dir = os.path.join(root_dir, 'annotations')
        self.transform = transform
        
        # Get all JSONs in the filtered directory
        self.annotation_files = glob.glob(os.path.join(self.annotation_dir, '*.json'))
        print(f"Dataset from {root_dir} initialized with {len(self.annotation_files)} samples.")

    def __len__(self):
        return len(self.annotation_files)

    def __getitem__(self, idx):
        ann_path = self.annotation_files[idx]
        
        try:
            with open(ann_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {ann_path}: {e}")
            return torch.zeros(3, IMG_SIZE[0], IMG_SIZE[1]), torch.zeros(1, IMG_SIZE[0], IMG_SIZE[1])

        base_name = os.path.splitext(os.path.basename(ann_path))[0]
        img_path = os.path.join(self.image_dir, base_name + '.jpg')
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return torch.zeros(3, IMG_SIZE[0], IMG_SIZE[1]), torch.zeros(1, IMG_SIZE[0], IMG_SIZE[1])

        w, h = image.size
        
        mask = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask)
        
        # Only draw masks for target labels
        for obj in data.get('objects', []):
            if is_target_label(obj['label']):
                bbox = obj['bbox']
                xmin = max(0, min(bbox['xmin'], w))
                ymin = max(0, min(bbox['ymin'], h))
                xmax = max(0, min(bbox['xmax'], w))
                ymax = max(0, min(bbox['ymax'], h))
                
                if xmax > xmin and ymax > ymin:
                    draw.rectangle(
                        [xmin, ymin, xmax, ymax], 
                        fill=255
                    )
        
        image = TF.resize(image, IMG_SIZE)
        mask = TF.resize(mask, IMG_SIZE, interpolation=transforms.InterpolationMode.NEAREST)
        
        image = TF.to_tensor(image)
        mask = torch.from_numpy(np.array(mask)).float().unsqueeze(0) / 255.0
        
        return image, mask


def evaluate():
    print(f"Using device: {DEVICE}")
    
    # Load Test Dataset
    test_dataset = FilteredTrafficSignDataset(TEST_DIR)
    
    if len(test_dataset) == 0:
        print("No test samples found for evaluation.")
        return

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0) # num_workers=0 for debugging ease

    # Load Model
    print(f"Loading model from {MODEL_PATH}...")
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file {MODEL_PATH} not found. Train the model first.")
        return

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        print(f"Error loading model state dict: {e}")
        return
        
    model.eval()
    
    total_acc = 0
    total_iou = 0
    num_samples = 0
    
    print(f"Evaluating on {len(test_dataset)} samples from {TEST_DIR}...")
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            
            outputs = model(images)
            
            acc, iou = calculate_metrics(outputs, masks)
            
            total_acc += acc * images.size(0) # Multiply by batch size to get sum
            total_iou += iou * images.size(0)
            num_samples += images.size(0)

    if num_samples > 0:
        avg_acc = total_acc / num_samples
        avg_iou = total_iou / num_samples
        print(f"\n--- Evaluation Results ---")
        print(f"Average Pixel Accuracy: {avg_acc:.4f}")
        print(f"Average IoU: {avg_iou:.4f}")
        print(f"Evaluated on {num_samples} samples.")
    else:
        print("No samples were evaluated. Check dataset and loader.")

if __name__ == '__main__':
    evaluate()