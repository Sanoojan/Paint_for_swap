import os
import yaml
from natsort import natsorted

# Paths to your folders
subfolders_root = "dataset/FaceData/Data/VFHQ-Test/GT/Vid_Interval1_512x512_LANCZOS4"        # Folder containing subfolders
source_images_root = "dataset/FaceData/Data/VFHQ-Test/Celeb_Source"  # Folder containing source images
output_yaml_path = "dataset/FaceData/Data/VFHQ-Test/data_matching.yaml"

# Get sorted list of subfolders and source images
subfolders = natsorted([f for f in os.listdir(subfolders_root) if os.path.isdir(os.path.join(subfolders_root, f))])
source_images = natsorted([f for f in os.listdir(source_images_root) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

assert len(subfolders) == len(source_images), "Mismatch between subfolders and source images!"

# Build the matching dictionary
matching = {}
for subfolder, image_name in zip(subfolders, source_images):
    matching[subfolder] = image_name

# Save to a single YAML file
with open(output_yaml_path, 'w') as f:
    yaml.dump(matching, f)

print(f"Matching YAML saved to {output_yaml_path}")
