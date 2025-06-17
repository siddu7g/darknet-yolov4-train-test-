import os
import random
import shutil

# === CONFIGURE THESE PATHS ===
images_dir = "/home/sidg/Delta_falsewing/all_data"
labels_dir = "/home/sidg/Downloads/labels_my-project-name_2025-06-16-10-42-23"
split_index = 452  # Number of items for training

train_dir = "/home/sidg/Delta_falsewing/falsewing_dataset/train"
val_dir = "/home/sidg/Delta_falsewing/falsewing_dataset/val"
output_dir = "/home/sidg/Delta_falsewing/falsewing_dataset"

# Make sure output directories exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# === Helper function to extract digits from filename for matching ===
def normalize(name):
    return ''.join(filter(str.isdigit, name))

# === Get sorted lists of image and label files ===
image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".jpg")])
label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith(".txt")])

# === Build dicts mapping normalized keys to filenames ===
image_map = {normalize(os.path.splitext(f)[0]): f for f in image_files}
label_map = {normalize(os.path.splitext(f)[0]): f for f in label_files}

# Find common keys to match image-label pairs
common_keys = sorted(set(image_map.keys()).intersection(label_map.keys()))
print(f"Found {len(common_keys)} matched image-label pairs.")

# === Split keys into train and val sets, then shuffle ===
train_keys = common_keys[:split_index]
val_keys = common_keys[split_index:]

random.seed(42)  # for reproducibility
random.shuffle(train_keys)
random.shuffle(val_keys)

# === Function to copy image-label pairs to a target directory ===
def copy_pairs(keys, target_dir):
    for key in keys:
        img_file = image_map[key]
        lbl_file = label_map[key]
        shutil.copy(os.path.join(images_dir, img_file), os.path.join(target_dir, img_file))
        shutil.copy(os.path.join(labels_dir, lbl_file), os.path.join(target_dir, lbl_file))

# Copy train and val pairs
copy_pairs(train_keys, train_dir)
copy_pairs(val_keys, val_dir)

print(f"✅ Copied {len(train_keys)} pairs to {train_dir}/")
print(f"✅ Copied {len(val_keys)} pairs to {val_dir}/")

# === Extract and combine all label text contents into a single .txt file ===
def extract_texts(from_dir, output_file):
    with open(output_file, 'w') as out_f:
        for file in sorted(os.listdir(from_dir)):
            if file.endswith(".txt"):
                with open(os.path.join(from_dir, file), 'r') as in_f:
                    content = in_f.read().strip()
                    if content:
                        out_f.write(content + '\n')

# Generate combined label text files
extract_texts(train_dir, os.path.join(output_dir, "train_text.txt"))
extract_texts(val_dir, os.path.join(output_dir, "val_text.txt"))

print("✅ Generated combined train_text.txt and val_text.txt in output/")

# === Write image paths (only .jpg) into train_dir.txt and val_dir.txt ===
def write_image_paths(images_folder, output_file):
    with open(output_file, 'w') as f:
        for filename in sorted(os.listdir(images_folder)):
            if filename.endswith(".jpg"):
                full_path = os.path.abspath(os.path.join(images_folder, filename))
                f.write(full_path + '\n')

write_image_paths(train_dir, os.path.join(output_dir, "train_dir.txt"))
write_image_paths(val_dir, os.path.join(output_dir, "val_dir.txt"))

print("✅ Generated train_dir.txt and val_dir.txt with image paths in output/")
