import glob
import json
import math
import os
import random
import shutil
from collections import defaultdict

# Configuration
DATA_DIR = "filtered_data"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
ANNOTATIONS_DIR = os.path.join(DATA_DIR, "annotations")

# Ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Stratification Categories
# Map keywords to high-level categories for balancing
CATEGORY_KEYWORDS = {
    'regulatory--keep-right': 'keepRight',
    'complementary--keep-right': 'keepRight',
    'warning--traffic-merges': 'merge',
    'information--pedestrians-crossing': 'pedestrian',
    'warning--pedestrians-crossing': 'pedestrian',
    'warning--traffic-signals': 'signal',
    'regulatory--stop': 'stop',
    'regulatory--yield': 'yield',
    'regulatory--maximum-speed-limit': 'speedLimit'
}

def get_category(label):
    for keyword, category in CATEGORY_KEYWORDS.items():
        if keyword in label:
            return category
    return None


def main():
    # 1. Inventory Data
    print("Inventorying data...")
    ann_files = glob.glob(os.path.join(ANNOTATIONS_DIR, "*.json"))

    # Map: image_filename -> set of categories present
    img_categories = defaultdict(set)
    category_counts = defaultdict(int)

    valid_files = []

    for ann_path in ann_files:
        try:
            with open(ann_path, "r") as f:
                data = json.load(f)

            cats = set()
            for obj in data.get("objects", []):
                cat = get_category(obj["label"])
                if cat:
                    cats.add(cat)

            if cats:
                base_name = os.path.basename(ann_path)
                # Check if image exists
                img_name = os.path.splitext(base_name)[0] + ".jpg"
                if os.path.exists(os.path.join(IMAGES_DIR, img_name)):
                    img_categories[base_name] = cats
                    valid_files.append(base_name)
                    for c in cats:
                        category_counts[c] += 1
        except Exception as e:
            print(f"Error reading {ann_path}: {e}")

    print(f"Found {len(valid_files)} valid samples.")
    print("Category Counts:", dict(category_counts))

    # 2. Stratified Split Logic
    # We assign each image to its 'rarest' category to ensure rare classes are covered.
    # Sort categories by frequency (ascending)
    sorted_cats = sorted(category_counts.keys(), key=lambda k: category_counts[k])

    # Buckets for each category (processed in order of rarity)
    cat_buckets = defaultdict(list)
    assigned_files = set()

    for cat in sorted_cats:
        for fname in valid_files:
            if fname in assigned_files:
                continue
            if cat in img_categories[fname]:
                cat_buckets[cat].append(fname)
                assigned_files.add(fname)

    # Now split each bucket
    splits = {"train": [], "val": [], "test": []}

    for cat, files in cat_buckets.items():
        random.shuffle(files)
        n = len(files)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        # Remainder goes to test

        splits["train"].extend(files[:n_train])
        splits["val"].extend(files[n_train : n_train + n_val])
        splits["test"].extend(files[n_train + n_val :])

    print(
        f"Split Results: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}"
    )

    # 3. Move Files
    for split_name in ["train", "val", "test"]:
        split_dir = os.path.join(DATA_DIR, split_name)
        split_img_dir = os.path.join(split_dir, "images")
        split_ann_dir = os.path.join(split_dir, "annotations")

        os.makedirs(split_img_dir, exist_ok=True)
        os.makedirs(split_ann_dir, exist_ok=True)

        print(f"Moving files to {split_name}...")
        for fname in splits[split_name]:
            # Move Annotation
            src_ann = os.path.join(ANNOTATIONS_DIR, fname)
            dst_ann = os.path.join(split_ann_dir, fname)
            shutil.move(src_ann, dst_ann)

            # Move Image
            img_name = os.path.splitext(fname)[0] + ".jpg"
            src_img = os.path.join(IMAGES_DIR, img_name)
            dst_img = os.path.join(split_img_dir, img_name)
            shutil.move(src_img, dst_img)

    # Cleanup original dirs if empty
    try:
        os.rmdir(IMAGES_DIR)
        os.rmdir(ANNOTATIONS_DIR)
    except:
        print("Original directories not empty, kept them.")

    print("Done.")


if __name__ == "__main__":
    random.seed(42)  # Reproducibility
    main()
