import os
import json
import glob
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms.functional as TF
from model import UNet

# Configuration
TEST_IMAGES_DIR = 'filtered_data/test/images'
TEST_ANNOTATIONS_DIR = 'filtered_data/test/annotations'
MODEL_PATH = 'models/model_filtered.pth' # Use the newly trained model
IMG_SIZE = (512, 512) # Consistent with training
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = 'test_results_filtered' # New output directory for filtered data results

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

def process_image_and_visualize(model, image_path, annotation_path, output_path):
    # Load Image
    try:
        pil_img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Could not open image {image_path}: {e}")
        return

    original_w, original_h = pil_img.size
    
    # Preprocess for Model
    input_img = TF.resize(pil_img, IMG_SIZE)
    input_tensor = TF.to_tensor(input_img).unsqueeze(0).to(DEVICE)
    
    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.sigmoid(output)
        # Create binary mask (0 or 1)
        mask = (probs > 0.5).float().cpu().numpy().squeeze()
    
    # Resize predicted mask back to original image size
    mask_uint8 = (mask * 255).astype(np.uint8)
    mask_resized_pred = cv2.resize(mask_uint8, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    
    # Create Ground Truth Mask
    gt_mask_pil = Image.new('L', (original_w, original_h), 0)
    draw_gt = ImageDraw.Draw(gt_mask_pil)
    
    try:
        with open(annotation_path, 'r') as f:
            data = json.load(f)
        for obj in data.get('objects', []):
            if is_target_label(obj['label']):
                bbox = obj['bbox']
                xmin = max(0, min(bbox['xmin'], original_w))
                ymin = max(0, min(bbox['ymin'], original_h))
                xmax = max(0, min(bbox['xmax'], original_w))
                ymax = max(0, min(bbox['ymax'], original_h))
                if xmax > xmin and ymax > ymin:
                    draw_gt.rectangle([xmin, ymin, xmax, ymax], fill=255)
    except Exception as e:
        print(f"Error loading or drawing ground truth for {annotation_path}: {e}")

    gt_mask_np = np.array(gt_mask_pil)
    
    # Convert PIL image to OpenCV format (BGR) for drawing
    opencv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    # Find Contours for Predicted Mask
    contours_pred, _ = cv2.findContours(mask_resized_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Find Contours for Ground Truth Mask
    contours_gt, _ = cv2.findContours(gt_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw Bounding Boxes
    # Draw Ground Truth in Green
    for cnt in contours_gt:
        if cv2.contourArea(cnt) < 100: continue # Filter small noise
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(opencv_img, (x, y), (x + w, y + h), (0, 255, 0), 2) # Green for GT
        
    # Draw Predictions in Red (if they don't perfectly overlap GT, will be visible)
    for cnt in contours_pred:
        if cv2.contourArea(cnt) < 100: continue # Filter small noise
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(opencv_img, (x, y), (x + w, y + h), (0, 0, 255), 2) # Red for Prediction
        
    # Save result
    cv2.imwrite(output_path, opencv_img)
    print(f"Saved visualization to {output_path}")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

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
    
    # Get all image files in the test directory
    image_files = glob.glob(os.path.join(TEST_IMAGES_DIR, '*.jpg'))
    
    if not image_files:
        print(f"No images found in {TEST_IMAGES_DIR}.")
        return

    print(f"Found {len(image_files)} images in test set. Processing for visualization...")
    
    for i, img_path in enumerate(image_files):
        base_name = os.path.basename(img_path)
        image_id = os.path.splitext(base_name)[0]
        annotation_path = os.path.join(TEST_ANNOTATIONS_DIR, image_id + '.json')
        
        if not os.path.exists(annotation_path):
            print(f"Skipping {img_path}: Corresponding annotation not found at {annotation_path}")
            continue

        save_path = os.path.join(OUTPUT_DIR, f"visual_result_{base_name}")
        process_image_and_visualize(model, img_path, annotation_path, save_path)

if __name__ == '__main__':
    main()