import os

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from model import UNet

# Configuration

MODEL_PATH = "models/model_filtered.pth"
IMAGE_PATH = "extracted_data/images/f4yIq45sBppvCYJHIZLi2Q.jpg"
OUTPUT_PATH = "f4yIq45sBppvCYJHIZLi2Q_raw_mask.png"
IMG_SIZE = (512, 512)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print(f"Loading model from {MODEL_PATH}...")
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file {MODEL_PATH} not found.")
        return

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        print(f"Error loading state dict: {e}")
        return

    model.eval()

    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image file {IMAGE_PATH} not found.")
        return

    print(f"Processing image: {IMAGE_PATH}")
    image = Image.open(IMAGE_PATH).convert("RGB")
    original_size = image.size

    # Preprocess
    input_image = TF.resize(image, IMG_SIZE)
    input_tensor = TF.to_tensor(input_image).unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.sigmoid(output)
        mask = (probs > 0.5).float()

    # Postprocess mask
    mask_cpu = mask.squeeze().cpu().numpy()
    # Convert to 0-255 uint8
    mask_img = Image.fromarray((mask_cpu * 255).astype(np.uint8))
    # Resize back to original size
    mask_img = mask_img.resize(original_size)

    mask_img.save(OUTPUT_PATH)
    print(f"Raw mask saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
