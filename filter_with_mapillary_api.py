import os
import json
import glob
import shutil
import time
import requests
from urllib.parse import urlparse, parse_qs
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
MAPILLARY_ACCESS_TOKEN = os.environ.get('MAPILLARY_TOKEN') 
SOURCE_ANNOTATION_DIR = 'extracted_data/Labels_and_stuff/mtsd_v2_partially_annotated/annotations'
SOURCE_IMAGE_DIR = 'extracted_data/images'
DEST_DIR = 'us_data_api_filtered'
CACHE_FILE = 'mapillary_cache.json'
MAX_WORKERS = 50  # Number of parallel threads

# Target prefixes/substrings to check for (ignoring --g1 or other suffixes)
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

# Bounding Box for "Lower 48" US check
# (Min Lon, Min Lat, Max Lon, Max Lat)
US_BOUNDS = [
    (-124.848974, 24.396308, -66.934570, 49.384358) 
]

def has_target_label(json_data):
    """Checks if the annotation contains any of the target sign types."""
    for obj in json_data.get('objects', []):
        label = obj['label']
        for keyword in TARGET_KEYWORDS:
            if keyword in label:
                return True
    return False

def is_in_us_lower48(lon, lat):
    """Checks if a point is roughly inside the contiguous US."""
    for (min_lon, min_lat, max_lon, max_lat) in US_BOUNDS:
        if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
            return True
    return False

def get_new_id_from_legacy(legacy_image_id):
    """
    Resolves a legacy image ID to its new numeric ID from the web app redirect.
    Returns new_id (str) or None.
    """
    url = f"https://www.mapillary.com/app/?pKey={legacy_image_id}"
    
    try:
        response = requests.get(url, allow_redirects=False, timeout=10)
        
        if response.status_code == 302 and 'Location' in response.headers:
            redirect_url = response.headers['Location']
            parsed_url = urlparse(redirect_url)
            query_params = parse_qs(parsed_url.query)
            
            if 'pKey' in query_params:
                return query_params['pKey'][0]
    except Exception:
        pass # Suppress errors for cleaner output when multithreaded
        
    return None

def get_coords_from_graph_api(new_image_id, access_token):
    """
    Fetches geometry for a given new image ID from Mapillary Graph API v4.
    Returns [lon, lat] or None.
    """
    url = f"https://graph.mapillary.com/{new_image_id}"
    headers = {"Authorization": f"OAuth {access_token}"}
    params = {"fields": "id,geometry"}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx) 
        
        data = response.json()
        if 'geometry' in data and 'coordinates' in data['geometry']:
            return data['geometry']['coordinates'] # [lon, lat]
    except Exception:
        pass # Suppress errors for cleaner output when multithreaded
        
    return None

def resolve_and_get_coords(legacy_image_id, access_token):
    """Combines legacy ID resolution and Graph API coordinate fetching."""
    new_id = get_new_id_from_legacy(legacy_image_id)
    if new_id:
        return get_coords_from_graph_api(new_id, access_token)
    return None

def process_candidate(args):
    """
    Worker function to process a single candidate.
    args: (ann_path, image_id, cached_coords, access_token)
    Returns: (image_id, coords, is_new_fetch)
    """
    ann_path, image_id, cached_coords, access_token = args
    
    if cached_coords:
        return image_id, cached_coords, False
    
    # If not in cache, fetch
    coords = resolve_and_get_coords(image_id, access_token)
    return image_id, coords, True

def main():
    if not MAPILLARY_ACCESS_TOKEN:
        print("Error: MAPILLARY_TOKEN environment variable not set. This is required for Graph API calls.")
        return

    # Create Dest Dirs
    os.makedirs(os.path.join(DEST_DIR, 'images'), exist_ok=True)
    os.makedirs(os.path.join(DEST_DIR, 'annotations'), exist_ok=True)

    # Load Cache
    cache = {}
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                cache = json.load(f)
            print(f"[INFO] Loaded {len(cache)} entries from cache.")
        except json.JSONDecodeError:
            print("[WARN] Cache file corrupted or empty, starting fresh.")

    # Scan Files
    all_json_files = glob.glob(os.path.join(SOURCE_ANNOTATION_DIR, '*.json'))
    print(f"[INFO] Scanning {len(all_json_files)} total annotations for candidates...")

    # Pre-filter list locally based on labels
    candidates = [] # list of (ann_path, image_id)
    for ann_path in all_json_files:
        try:
            with open(ann_path, 'r') as f:
                data = json.load(f)
            if has_target_label(data):
                image_id = os.path.splitext(os.path.basename(ann_path))[0]
                candidates.append((ann_path, image_id))
        except Exception:
            pass
            
    print(f"[INFO] Found {len(candidates)} candidates containing target labels. Processing with {MAX_WORKERS} threads...")

    count_us = 0
    newly_cached = 0
    
    # Prepare tasks
    tasks = []
    for ann_path, image_id in candidates:
        cached_coords = cache.get(image_id)
        tasks.append((ann_path, image_id, cached_coords, MAPILLARY_ACCESS_TOKEN))

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_candidate = {executor.submit(process_candidate, task): task for task in tasks}
        
        for i, future in enumerate(as_completed(future_to_candidate)):
            ann_path, image_id, _, _ = future_to_candidate[future] # Extract original task data
            
            try:
                result_id, coords, is_new = future.result()
                
                if is_new:
                    cache[result_id] = coords
                    newly_cached += 1
                
                if coords:
                    lon, lat = coords
                    if is_in_us_lower48(lon, lat):
                        # Copy Files (Sequential I/O to avoid race conditions or disk thrashing)
                        dest_ann = os.path.join(DEST_DIR, 'annotations', os.path.basename(ann_path))
                        src_img = os.path.join(SOURCE_IMAGE_DIR, image_id + '.jpg')
                        dest_img = os.path.join(DEST_DIR, 'images', image_id + '.jpg')
                        
                        if os.path.exists(src_img):
                            shutil.copy2(ann_path, dest_ann)
                            shutil.copy2(src_img, dest_img)
                            count_us += 1

            except Exception as e:
                print(f"[ERROR] Exception processing {image_id}: {e}")
            
            # Periodic status and cache save
            if (i + 1) % 100 == 0:
                print(f"[INFO] Processed {i + 1}/{len(candidates)}... (US Found: {count_us})")
                if newly_cached > 0:
                    with open(CACHE_FILE, 'w') as f:
                        json.dump(cache, f)
                    newly_cached = 0

    # Final Cache Save
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"\n[INFO] Done in {duration:.2f} seconds.")
    print(f"[INFO] Processed {len(candidates)} candidates. Found {count_us} in US Lower 48.")
    print(f"[INFO] Filtered data saved to {DEST_DIR}")

if __name__ == '__main__':
    main()
