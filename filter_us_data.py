import os
import json
import glob
import shutil

# Configuration
SOURCE_ANNOTATION_DIR = 'extracted_data/Labels_and_stuff/mtsd_v2_partially_annotated/annotations'
SOURCE_IMAGE_DIR = 'extracted_data/images'
DEST_DIR = 'filtered_data'
DEST_ANNOTATION_DIR = os.path.join(DEST_DIR, 'annotations')
DEST_IMAGE_DIR = os.path.join(DEST_DIR, 'images')

# Target Categories (exact matches or prefixes)
TARGET_LABELS_EXACT = {
    'regulatory--keep-right--g1',
    'complementary--keep-right--g1',
    'warning--traffic-merges-left--g1',
    'warning--traffic-merges-right--g1',
    'information--pedestrians-crossing--g1',
    'warning--pedestrians-crossing--g1',
    'warning--traffic-signals--g1',
    'regulatory--stop--g1',
    'regulatory--yield--g1'
}

# For Speed Limits, we check prefix
SPEED_LIMIT_PREFIX = 'regulatory--maximum-speed-limit-'

def is_target_label(label):
    # Check for exact matches
    if label in TARGET_LABELS_EXACT:
        return True
    
    # Check for Speed Limits (ensure it is a US g1 sign)
    if label.startswith(SPEED_LIMIT_PREFIX) and label.endswith('--g1'):
        return True
        
    return False

def main():
    # 1. Create Destination Directories
    if not os.path.exists(DEST_ANNOTATION_DIR):
        os.makedirs(DEST_ANNOTATION_DIR)
    if not os.path.exists(DEST_IMAGE_DIR):
        os.makedirs(DEST_IMAGE_DIR)

    print(f"Scanning {SOURCE_ANNOTATION_DIR}...")
    
    json_files = glob.glob(os.path.join(SOURCE_ANNOTATION_DIR, '*.json'))
    
    count_copied = 0
    files_with_missing_images = 0
    
    for ann_path in json_files:
        try:
            with open(ann_path, 'r') as f:
                data = json.load(f)
            
            # Check if the annotation contains any relevant object
            has_target_object = False
            for obj in data.get('objects', []):
                if is_target_label(obj['label']):
                    has_target_object = True
                    break
            
            if has_target_object:
                # Prepare paths
                base_name = os.path.basename(ann_path)
                image_name = os.path.splitext(base_name)[0] + '.jpg'
                source_image_path = os.path.join(SOURCE_IMAGE_DIR, image_name)
                
                # Check if image exists
                if os.path.exists(source_image_path):
                    # Copy Annotation
                    dest_ann_path = os.path.join(DEST_ANNOTATION_DIR, base_name)
                    shutil.copy2(ann_path, dest_ann_path)
                    
                    # Copy Image
                    dest_image_path = os.path.join(DEST_IMAGE_DIR, image_name)
                    shutil.copy2(source_image_path, dest_image_path)
                    
                    count_copied += 1
                else:
                    files_with_missing_images += 1
                    
        except Exception as e:
            print(f"Error processing {ann_path}: {e}")

    print("-" * 30)
    print(f"Processing Complete.")
    print(f"Total files copied: {count_copied}")
    if files_with_missing_images > 0:
        print(f"Skipped {files_with_missing_images} annotations because the image file was missing.")
    print(f"Filtered data located in: {DEST_DIR}")
    print("Note: 'yield-ahead' signs were not found in the dataset and are excluded.")
    print("Note: Filtered for US signs using '--g1' suffix as GPS coordinates were unavailable in annotation files.")

if __name__ == '__main__':
    main()
