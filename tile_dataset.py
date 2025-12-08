import os
import json
import glob
import shutil
import numpy as np
from PIL import Image

# Configuration
SOURCE_ROOT = 'filtered_data'
DEST_ROOT = 'tiled_data'
TILE_SIZE = 512
MIN_VISIBILITY = 0.3 # Keep object if at least 30% of it is inside the tile

def process_image(img_path, ann_path, dest_img_dir, dest_ann_dir):
    try:
        im = Image.open(img_path).convert("RGB")
        w, h = im.size
        
        with open(ann_path, 'r') as f:
            ann_data = json.load(f)
        
        objects = ann_data.get('objects', [])
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # Calculate tiles
        # We'll just step through.
        # Ideally we want some overlap to avoid cutting signs, but let's start simple.
        stride = TILE_SIZE # No overlap
        
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                # Adjust last tile to not go out of bounds? 
                # Or just pad? Let's crop. 
                # If (x+TILE_SIZE) > w, we can either skip or shift back.
                # Shift back strategy:
                real_x = x
                real_y = y
                if real_x + TILE_SIZE > w: real_x = w - TILE_SIZE
                if real_y + TILE_SIZE > h: real_y = h - TILE_SIZE
                
                # Ensure we don't have negative coords (image smaller than tile)
                if real_x < 0: real_x = 0
                if real_y < 0: real_y = 0
                
                # Crop box
                tile_box = (real_x, real_y, real_x + TILE_SIZE, real_y + TILE_SIZE)
                tile_im = im.crop(tile_box)
                
                # Filter objects for this tile
                tile_objects = []
                for obj in objects:
                    bbox = obj['bbox']
                    # Check intersection
                    bx1, by1, bx2, by2 = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']
                    
                    # Object box in image coords
                    ox1 = max(bx1, real_x)
                    oy1 = max(by1, real_y)
                    ox2 = min(bx2, real_x + TILE_SIZE)
                    oy2 = min(by2, real_y + TILE_SIZE)
                    
                    if ox2 > ox1 and oy2 > oy1:
                        # Intersection exists
                        inter_area = (ox2 - ox1) * (oy2 - oy1)
                        obj_area = (bx2 - bx1) * (by2 - by1)
                        
                        if (inter_area / obj_area) >= MIN_VISIBILITY:
                            # Keep object, adjust coords relative to tile
                            new_obj = obj.copy()
                            new_obj['bbox'] = {
                                'xmin': ox1 - real_x,
                                'ymin': oy1 - real_y,
                                'xmax': ox2 - real_x,
                                'ymax': oy2 - real_y
                            }
                            tile_objects.append(new_obj)
                
                # Save tile ONLY if it has objects (to solve imbalance)
                if len(tile_objects) > 0:
                    tile_filename = f"{base_name}_{real_x}_{real_y}"
                    
                    # Save Image
                    tile_im.save(os.path.join(dest_img_dir, tile_filename + '.jpg'))
                    
                    # Save Annotation
                    new_ann = ann_data.copy()
                    new_ann['width'] = TILE_SIZE
                    new_ann['height'] = TILE_SIZE
                    new_ann['objects'] = tile_objects
                    
                    with open(os.path.join(dest_ann_dir, tile_filename + '.json'), 'w') as f:
                        json.dump(new_ann, f)

    except Exception as e:
        print(f"Error processing {img_path}: {e}")

def main():
    if os.path.exists(DEST_ROOT):
        shutil.rmtree(DEST_ROOT)
    
    for split in ['train', 'val', 'test']:
        print(f"Processing {split}...")
        src_dir = os.path.join(SOURCE_ROOT, split)
        dst_dir = os.path.join(DEST_ROOT, split)
        
        src_img_dir = os.path.join(src_dir, 'images')
        src_ann_dir = os.path.join(src_dir, 'annotations')
        
        dst_img_dir = os.path.join(dst_dir, 'images')
        dst_ann_dir = os.path.join(dst_dir, 'annotations')
        
        os.makedirs(dst_img_dir)
        os.makedirs(dst_ann_dir)
        
        ann_files = glob.glob(os.path.join(src_ann_dir, '*.json'))
        
        for ann_path in ann_files:
            base_name = os.path.splitext(os.path.basename(ann_path))[0]
            img_path = os.path.join(src_img_dir, base_name + '.jpg')
            
            if os.path.exists(img_path):
                process_image(img_path, ann_path, dst_img_dir, dst_ann_dir)

    print("Tiling complete.")

if __name__ == '__main__':
    main()
