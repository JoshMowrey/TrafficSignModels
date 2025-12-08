import os
import zipfile

source_dir = "Compressed data"
dest_dir = "extracted_data"

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

for filename in os.listdir(source_dir):
    file_path = os.path.join(source_dir, filename)

    if os.path.isfile(file_path):
        print(f"Processing: {filename}")
        if zipfile.is_zipfile(file_path):
            try:
                with zipfile.ZipFile(file_path, "r") as zip_ref:
                    zip_ref.extractall(dest_dir)
                print(f"Extracted: {filename}")
            except zipfile.BadZipFile:
                print(f"Error: Bad zip file - {filename}")
            except Exception as e:
                print(f"Error extracting {filename}: {e}")
        else:
            print(f"Skipped (not a zip file): {filename}")
