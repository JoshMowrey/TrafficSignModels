import os
import json
import glob
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader

# --- CONFIGURATION ---
MODELS_DIR = 'models'
TEST_DIR = 'filtered_data/test'
OUTPUT_DIR = 'benchmark_results'
IMG_SIZE = (512, 512)
BATCH_SIZE = 1 # Must be 1 for per-sample metadata handling
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Map keywords to readable Categories
CATEGORY_MAP = {
    'regulatory--stop': 'Stop',
    'regulatory--yield': 'Yield',
    'regulatory--keep-right': 'Keep Right', 
    'complementary--keep-right': 'Keep Right',
    'warning--traffic-merges': 'Merge', 
    'information--pedestrians-crossing': 'Pedestrian',
    'warning--pedestrians-crossing': 'Pedestrian', 
    'warning--traffic-signals': 'Traffic Signal',
    'regulatory--maximum-speed-limit': 'Speed Limit'
}

TARGET_KEYWORDS = list(CATEGORY_MAP.keys())

# --- ARCHITECTURES ---

# 1. New Residual Architecture (32 base channels)
class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout_prob=0.1):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_prob),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
        )
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, padding=0),
                nn.BatchNorm2d(out_c)
            )
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv_block(x)
        out += residual
        out = self.relu(out)
        return out

class NewUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=32):
        super(NewUNet, self).__init__()
        c1, c2, c3, c4 = base_channels, base_channels*2, base_channels*4, base_channels*8
        self.dconv_down1 = ResidualBlock(in_channels, c1)
        self.dconv_down2 = ResidualBlock(c1, c2)
        self.dconv_down3 = ResidualBlock(c2, c3)
        self.dconv_down4 = ResidualBlock(c3, c4)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dconv_up3 = ResidualBlock(c3 + c4, c3)
        self.dconv_up2 = ResidualBlock(c2 + c3, c2)
        self.dconv_up1 = ResidualBlock(c1 + c2, c1)
        self.conv_last = nn.Conv2d(c1, out_channels, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        x = self.dconv_down4(x)
        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        return self.conv_last(x)

# 2. Old Standard Architecture (64 base channels)
class OldUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(OldUNet, self).__init__()
        def double_conv(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)
            )
        self.dconv_down1 = double_conv(in_channels, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(64 + 128, 64)
        self.conv_last = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        x = self.dconv_down4(x)
        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        return self.conv_last(x)

# --- HELPERS ---

def get_category(label):
    for keyword, category in CATEGORY_MAP.items():
        if keyword in label:
            return category
    return None

def is_target_label(label):
    return get_category(label) is not None

def calculate_metrics(pred, target, threshold=0.5):
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)
    
    correct = (pred_flat == target_flat).float().sum()
    accuracy = correct / target_flat.numel()
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    iou = 1.0 if union == 0 else intersection / union
    return accuracy.item(), iou.item()

class FilteredTrafficSignDataset(Dataset):
    def __init__(self, root_dir):
        self.image_dir = os.path.join(root_dir, 'images')
        self.annotation_dir = os.path.join(root_dir, 'annotations')
        self.annotation_files = glob.glob(os.path.join(self.annotation_dir, '*.json'))

    def __len__(self):
        return len(self.annotation_files)

    def __getitem__(self, idx):
        ann_path = self.annotation_files[idx]
        categories = []
        try:
            with open(ann_path, 'r') as f:
                data = json.load(f)
        except:
            return torch.zeros(3, IMG_SIZE[0], IMG_SIZE[1]), torch.zeros(1, IMG_SIZE[0], IMG_SIZE[1]), []

        base_name = os.path.splitext(os.path.basename(ann_path))[0]
        img_path = os.path.join(self.image_dir, base_name + '.jpg')
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            return torch.zeros(3, IMG_SIZE[0], IMG_SIZE[1]), torch.zeros(1, IMG_SIZE[0], IMG_SIZE[1]), []

        w, h = image.size
        mask = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask)
        
        for obj in data.get('objects', []):
            cat = get_category(obj['label'])
            if cat:
                categories.append(cat)
                bbox = obj['bbox']
                xmin = max(0, min(bbox['xmin'], w))
                ymin = max(0, min(bbox['ymin'], h))
                xmax = max(0, min(bbox['xmax'], w))
                ymax = max(0, min(bbox['ymax'], h))
                if xmax > xmin and ymax > ymin:
                    draw.rectangle([xmin, ymin, xmax, ymax], fill=255)
        
        image = TF.resize(image, IMG_SIZE)
        mask = TF.resize(mask, IMG_SIZE, interpolation=0) # Nearest
        # Return categories list as well
        return TF.to_tensor(image), torch.from_numpy(np.array(mask)).float().unsqueeze(0) / 255.0, list(set(categories))

# 3. EfficientNet-B0 U-Net Architecture (Fixed)
import torchvision.models as models
import torch.nn.functional as F

class EfficientNetUNetFixed(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, pretrained=False): # Pretrained False for loading state dict
        super(EfficientNetUNetFixed, self).__init__()
        weights = None # No weights needed for loading state dict
        self.encoder = models.efficientnet_b0(weights=weights)
        self.filters = [16, 24, 40, 112, 320] 
        self.up4 = self._make_up_block(self.filters[4], self.filters[3], self.filters[3])
        self.up3 = self._make_up_block(self.filters[3], self.filters[2], self.filters[2])
        self.up2 = self._make_up_block(self.filters[2], self.filters[1], self.filters[1])
        self.up1 = self._make_up_block(self.filters[1], self.filters[0], self.filters[0])
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.filters[0], 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1)
        )
    def _make_up_block(self, in_c, skip_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c + skip_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)
        )
    def forward(self, x):
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        features = self.encoder.features
        x1 = features[1](features[0](x))
        x2 = features[2](x1)
        x3 = features[3](x2)
        x4 = features[5](features[4](x3))
        x5 = features[7](features[6](x4))
        u4 = F.interpolate(x5, size=x4.shape[2:], mode='bilinear', align_corners=True)
        d4 = self.up4(torch.cat([u4, x4], dim=1))
        u3 = F.interpolate(d4, size=x3.shape[2:], mode='bilinear', align_corners=True)
        d3 = self.up3(torch.cat([u3, x3], dim=1))
        u2 = F.interpolate(d3, size=x2.shape[2:], mode='bilinear', align_corners=True)
        d2 = self.up2(torch.cat([u2, x2], dim=1))
        u1 = F.interpolate(d2, size=x1.shape[2:], mode='bilinear', align_corners=True)
        d1 = self.up1(torch.cat([u1, x1], dim=1))
        return self.final_up(d1)

def try_load_model(path):
    """Attempts to load a model trying all architectures."""
    state_dict = torch.load(path, map_location=DEVICE)
    
    # Try New Arch (Residual)
    model = NewUNet(in_channels=3, out_channels=1).to(DEVICE)
    try:
        model.load_state_dict(state_dict)
        print(f"  -> Loaded {os.path.basename(path)} as New Residual UNet")
        return model
    except RuntimeError:
        pass 

    # Try Old Arch (Standard)
    model = OldUNet(in_channels=3, out_channels=1).to(DEVICE)
    try:
        model.load_state_dict(state_dict)
        print(f"  -> Loaded {os.path.basename(path)} as Old Standard UNet")
        return model
    except RuntimeError:
        pass

    # Try EfficientNet Arch
    model = EfficientNetUNetFixed(in_channels=3, out_channels=1).to(DEVICE)
    try:
        model.load_state_dict(state_dict)
        print(f"  -> Loaded {os.path.basename(path)} as EfficientNet-B0 UNet")
        return model
    except RuntimeError:
        print(f"  -> Failed to load {os.path.basename(path)} with known architectures.")
        return None

def draw_text(img, text, pos=(10, 30), color=(255, 255, 255)):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

def custom_collate(batch):
    """
    Custom collate function to handle variable-length category lists.
    Batch is a list of tuples: [(image, mask, categories), ...]
    """
    images = []
    masks = []
    categories_batch = []
    
    for img, mask, cats in batch:
        images.append(img)
        masks.append(mask)
        categories_batch.append(cats)
        
    return torch.stack(images), torch.stack(masks), categories_batch

# --- MAIN ---

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    print(f"Searching for models in {MODELS_DIR}...")
    model_files = glob.glob(os.path.join(MODELS_DIR, '*.pth'))
    loaded_models = {}

    for f in model_files:
        model = try_load_model(f)
        if model:
            model.eval()
            loaded_models[os.path.basename(f)] = model

    if not loaded_models:
        print("No valid models loaded.")
        return

    # Setup Data
    dataset = FilteredTrafficSignDataset(TEST_DIR)
    if len(dataset) == 0:
        print("Test dataset empty.")
        return
    
    # Use custom collate_fn
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=custom_collate)
    
    print("-" * 60)
    print(f"EVALUATION METRICS (Per Category)")
    print("-" * 60)

    # Evaluate each model
    model_stats = {} 

    criterion = nn.BCEWithLogitsLoss()
    
    # Unique Categories found in dataset (for table headers)
    all_categories = sorted(list(set(CATEGORY_MAP.values())))

    for name, model in loaded_models.items():
        total_loss = 0
        total_acc = 0
        total_iou = 0
        n_samples = 0
        
        # Per-category trackers
        # cat_stats = {'Stop': {'iou': sum, 'count': n}, ...}
        cat_stats = {cat: {'iou': 0.0, 'acc': 0.0, 'count': 0} for cat in all_categories}
        
        with torch.no_grad():
            for images, masks, categories_batch in loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)
                
                out = model(images)
                loss = criterion(out, masks)
                
                acc, iou = calculate_metrics(out, masks)
                
                total_loss += loss.item()
                total_acc += acc
                total_iou += iou
                n_samples += 1
                
                # Track per category (if multiple, count for all present)
                # categories_batch is a tuple of lists because batch_size=1
                # Actually, with batch_size=1, categories_batch is [('Stop',)] if collate didn't flatten.
                # Let's inspect how DataLoader returns lists of strings. 
                # It usually returns a list of tuples or list of lists.
                # Since batch_size=1, it will be a list of length 1 containing a tuple of categories?
                # Let's simplify: iterate the batch list
                
                # Extract categories from the batch (tuple/list structure from collate)
                # categories_batch is likely: [('Stop',), ('Yield',)] transposed.
                # Actually with default collate and batch_size=1, `categories_batch` is a list of tuples if return was list.
                # Let's just flatten.
                
                current_cats = []
                for c in categories_batch:
                    if len(c) > 0:
                         current_cats.append(c[0]) # Extract string from tuple
                
                for cat in current_cats:
                    if cat in cat_stats:
                        cat_stats[cat]['iou'] += iou
                        cat_stats[cat]['acc'] += acc
                        cat_stats[cat]['count'] += 1

        avg_loss = total_loss / n_samples
        avg_acc = total_acc / n_samples
        avg_iou = total_iou / n_samples
        
        model_stats[name] = (avg_acc, avg_iou)
        
        print(f"\nModel: {name}")
        print(f"{'Global':<15} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f} | IoU: {avg_iou:.4f}")
        print("-" * 45)
        print(f"{'Category':<15} | {'Count':<5} | {'Acc':<7} | {'IoU':<7}")
        print("-" * 45)
        
        for cat in all_categories:
            stats = cat_stats[cat]
            count = stats['count']
            if count > 0:
                c_acc = stats['acc'] / count
                c_iou = stats['iou'] / count
                print(f"{cat:<15} | {count:<5} | {c_acc:.4f}  | {c_iou:.4f}")
            else:
                print(f"{cat:<15} | 0     | N/A      | N/A")
        print("-" * 60)

    print("Generating Visualizations for 10 samples...")

    # Visualization
    # We need original images to draw on, so we can't just use the loader easily since it transforms them.
    # We'll load files manually for the first 10.
    
    sample_files = dataset.annotation_files[:10]
    
    colors = [(0, 0, 255), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255)] # Red, Blue, Cyan, Yellow, Magenta
    
    for ann_path in sample_files:
        base_name = os.path.basename(ann_path)
        img_name = os.path.splitext(base_name)[0] + '.jpg'
        img_path = os.path.join(dataset.image_dir, img_name)
        
        # Load Original for CV2
        pil_img = Image.open(img_path).convert("RGB")
        w, h = pil_img.size
        orig_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        # Draw Ground Truth (Green)
        try:
            with open(ann_path, 'r') as f:
                d = json.load(f)
            for obj in d.get('objects', []):
                if is_target_label(obj['label']):
                    bbox = obj['bbox']
                    pt1 = (int(bbox['xmin']), int(bbox['ymin']))
                    pt2 = (int(bbox['xmax']), int(bbox['ymax']))
                    cv2.rectangle(orig_cv, pt1, pt2, (0, 255, 0), 3)
        except: pass
        
        draw_text(orig_cv, "GT (Green)", (10, 30), (0, 255, 0))

        # Prepare input
        input_img = TF.resize(pil_img, IMG_SIZE)
        input_tensor = TF.to_tensor(input_img).unsqueeze(0).to(DEVICE)

        # Create a list of images to stack: [GT, Model1, Model2, ...]
        images_row = [orig_cv]
        
        for i, (name, model) in enumerate(loaded_models.items()):
            res_cv = orig_cv.copy()
            color = colors[i % len(colors)]
            
            with torch.no_grad():
                out = model(input_tensor)
                mask = (torch.sigmoid(out) > 0.5).float().cpu().numpy().squeeze()
            
            mask_uint8 = (mask * 255).astype(np.uint8)
            mask_resized = cv2.resize(mask_uint8, (w, h), interpolation=0)
            
            contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) < 50: continue
                x_box, y_box, w_box, h_box = cv2.boundingRect(cnt)
                cv2.rectangle(res_cv, (x_box, y_box), (x_box+w_box, y_box+h_box), color, 3)
            
            draw_text(res_cv, f"{name} ({model_stats[name][1]:.2f} IoU)", (10, 30), color)
            images_row.append(res_cv)

        # Concatenate
        target_h = 400
        resized_row = []
        for img in images_row:
            scale = target_h / img.shape[0]
            new_w = int(img.shape[1] * scale)
            resized_row.append(cv2.resize(img, (new_w, target_h)))
            
        combined = np.hstack(resized_row)
        out_file = os.path.join(OUTPUT_DIR, f"bench_{img_name}")
        cv2.imwrite(out_file, combined)
        print(f"Saved {out_file}")

    print("Done.")

if __name__ == '__main__':
    main()
