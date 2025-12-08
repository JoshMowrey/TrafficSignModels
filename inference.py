import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from model import UNet

# Configuration
MODEL_PATH = 'unet_traffic_sign.pth'
IMAGE_DIR = 'extracted_data/images'
IMG_SIZE = (256, 256)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict_single_image(image_path, save_path='inference_result.png'):
    print(f"Loading model from {MODEL_PATH}...")
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    
    if not os.path.exists(MODEL_PATH):
        print("Error: Model file not found. Train the model first.")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print(f"Processing image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    
    # Preprocess
    input_image = TF.resize(image, IMG_SIZE)
    input_tensor = TF.to_tensor(input_image).unsqueeze(0).to(DEVICE)
    
    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.sigmoid(output)
        mask = (probs > 0.5).float()
    
    # Postprocess mask to visualize
    mask_cpu = mask.squeeze().cpu().numpy()
    mask_img = Image.fromarray((mask_cpu * 255).astype(np.uint8))
    mask_img = mask_img.resize(original_size) # Resize back to original for comparison
    
    # Create side-by-side comparison
    result = Image.new('RGB', (original_size[0] * 2, original_size[1]))
    result.paste(image, (0, 0))
    result.paste(mask_img, (original_size[0], 0))
    
    result.save(save_path)
    print(f"Result saved to {save_path}")

if __name__ == '__main__':
    # Find a random image
    files = os.listdir(IMAGE_DIR)
    jpg_files = [f for f in files if f.endswith('.jpg')]
    
    if jpg_files:
        # Pick one
        target_image = os.path.join(IMAGE_DIR, jpg_files[0])
        predict_single_image(target_image)
    else:
        print("No images found to test.")
