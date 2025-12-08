import json
import glob
import os
from collections import Counter

def count_labels(annotations_dir):
    label_counts = Counter()
    json_files = glob.glob(os.path.join(annotations_dir, "*.json"))
    
    print(f"Found {len(json_files)} JSON files.")

    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if 'objects' in data:
                    for obj in data['objects']:
                        if 'label' in obj:
                            label_counts[obj['label']] += 1
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    return label_counts

if __name__ == "__main__":
    annotations_dir = "extracted_data/Labels_and_stuff/mtsd_v2_partially_annotated/annotations"
    counts = count_labels(annotations_dir)
    
    print("\nLabel Counts:")
    # Sort by count descending
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    
    for label, count in sorted_counts:
        print(f"{label}: {count}")

    print(f"\nTotal unique labels: {len(counts)}")
