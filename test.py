import os
import shutil
import random
from pathlib import Path

# ===== CONFIGURATION - CHANGE THIS =====
main_folder = "dataset"  # The folder containing 'images' and 'animals'
output_folder = "organized/dataset"  # Where to save organized data

# ===== STEP 1: CREATE FOLDER STRUCTURE =====
classes = ['cat', 'dog', 'panda']  # Your actual classes

for split in ['train', 'val', 'test']:
    for class_name in classes:
        os.makedirs(os.path.join(output_folder, split, class_name), exist_ok=True)

# ===== STEP 2: FIND ALL IMAGES EVERYWHERE =====
print("🔍 Scanning for images in ALL folders...")

image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.bmp', '.gif']
all_images = []

# Recursively find EVERY image file in the main folder
for ext in image_extensions:
    all_images.extend(Path(main_folder).rglob(f'*{ext}'))

print(f"📸 Total images found: {len(all_images)}")

# ===== STEP 3: CLASSIFY IMAGES BY FOLDER NAME =====
classified = {
    'cat': [],
    'dog': [],
    'panda': [],
    'unknown': []
}

# Keywords to help classification
cat_keywords = ['cat', 'cats', 'kitten']
dog_keywords = ['dog', 'dogs', 'puppy']
panda_keywords = ['panda', 'pandas', 'bear']

for img_path in all_images:
    # Convert path to string for searching
    path_str = str(img_path).lower()
    filename = img_path.name.lower()
    
    # Check which class this belongs to based on FOLDER NAME
    if any(keyword in path_str for keyword in cat_keywords):
        classified['cat'].append(img_path)
    elif any(keyword in path_str for keyword in dog_keywords):
        classified['dog'].append(img_path)
    elif any(keyword in path_str for keyword in panda_keywords):
        classified['panda'].append(img_path)
    else:
        classified['unknown'].append(img_path)

# Print classification results
print("\n📊 Classification Results:")
print(f"🐱 Cat images: {len(classified['cat'])}")
print(f"🐶 Dog images: {len(classified['dog'])}")
print(f"🐼 Panda images: {len(classified['panda'])}")
print(f"❓ Unknown images: {len(classified['unknown'])}")

# ===== STEP 4: SHOW WHERE IMAGES WERE FOUND =====
print("\n📍 Image locations:")
for class_name in ['cat', 'dog', 'panda']:
    if classified[class_name]:
        print(f"\n{class_name.upper()} images found in:")
        locations = set()
        for img in classified[class_name][:5]:  # Show first 5 examples
            parent = img.parent.name
            grandparent = img.parent.parent.name
            locations.add(f"  - .../{grandparent}/{parent}/")
        for loc in locations:
            print(loc)

# ===== STEP 5: HANDLE UNKNOWN IMAGES =====
if classified['unknown']:
    print(f"\n⚠️  Found {len(classified['unknown'])} unknown images!")
    unknown_folder = os.path.join(output_folder, '00_UNKNOWN')
    os.makedirs(unknown_folder, exist_ok=True)
    
    for img_path in classified['unknown']:
        shutil.copy(img_path, os.path.join(unknown_folder, img_path.name))
    
    print(f"✅ Unknown images copied to: {unknown_folder}")
    print("   Check these manually and move to correct cat/dog/panda folders!")

# ===== STEP 6: CHECK FOR PANDA VS BEAR CONFUSION =====
print("\n🔍 Checking for panda images...")
panda_folders = []
for img_path in classified['panda']:
    folder_name = img_path.parent.name.lower()
    if 'bear' in folder_name or 'red' in folder_name:
        print(f"⚠️  Warning: Panda image found in '{folder_name}' folder - verify this is actually a panda!")

# ===== STEP 7: CREATE TRAIN/VAL/TEST SPLIT =====
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

print("\n📁 Creating train/val/test split...")

for class_name in ['cat', 'dog', 'panda']:
    images = classified[class_name]
    
    if len(images) == 0:
        print(f"⚠️  No {class_name} images found!")
        continue
    
    # Shuffle images
    random.shuffle(images)
    
    # Calculate split indices
    n_images = len(images)
    n_train = int(n_images * train_ratio)
    n_val = int(n_images * (train_ratio + val_ratio))
    
    train_images = images[:n_train]
    val_images = images[n_train:n_val]
    test_images = images[n_val:]
    
    # Copy images to respective folders
    print(f"\n{class_name.upper()}:")
    
    for img_path in train_images:
        dest = os.path.join(output_folder, 'train', class_name, img_path.name)
        shutil.copy(img_path, dest)
    print(f"  ✅ Train: {len(train_images)} images")
    
    for img_path in val_images:
        dest = os.path.join(output_folder, 'val', class_name, img_path.name)
        shutil.copy(img_path, dest)
    print(f"  ✅ Val: {len(val_images)} images")
    
    for img_path in test_images:
        dest = os.path.join(output_folder, 'test', class_name, img_path.name)
        shutil.copy(img_path, dest)
    print(f"  ✅ Test: {len(test_images)} images")

# ===== STEP 8: SHOW FINAL STRUCTURE =====
print("\n" + "="*50)
print("✅ DATASET ORGANIZED SUCCESSFULLY!")
print("="*50)
print(f"\n📁 Output folder: {output_folder}")
print("\nFolder structure created:")
print(f"""
{output_folder}/
│
├── train/
│   ├── cat/     ({len(classified['cat']) * train_ratio:.0f} images)
│   ├── dog/     ({len(classified['dog']) * train_ratio:.0f} images)
│   └── panda/   ({len(classified['panda']) * train_ratio:.0f} images)
│
├── val/
│   ├── cat/     ({len(classified['cat']) * val_ratio:.0f} images)
│   ├── dog/     ({len(classified['dog']) * val_ratio:.0f} images)
│   └── panda/   ({len(classified['panda']) * val_ratio:.0f} images)
│
├── test/
│   ├── cat/     ({len(classified['cat']) * test_ratio:.0f} images)
│   ├── dog/     ({len(classified['dog']) * test_ratio:.0f} images)
│   └── panda/   ({len(classified['panda']) * test_ratio:.0f} images)
│
└── 00_UNKNOWN/  ({len(classified['unknown'])} images to check manually)
""")

# ===== STEP 9: CREATE SUMMARY REPORT =====
with open(os.path.join(output_folder, 'dataset_summary.txt'), 'w') as f:
    f.write("ANIMAL DATASET SUMMARY\n")
    f.write("="*50 + "\n\n")
    f.write(f"Total images processed: {len(all_images)}\n")
    f.write(f"Cats: {len(classified['cat'])}\n")
    f.write(f"Dogs: {len(classified['dog'])}\n")
    f.write(f"Pandas: {len(classified['panda'])}\n")
    f.write(f"Unknown: {len(classified['unknown'])}\n\n")
    f.write("Train/Val/Test split: 70/15/15\n")
    f.write(f"Output location: {output_folder}")

print(f"\n📝 Summary saved to: {os.path.join(output_folder, 'dataset_summary.txt')}")

# ===== STEP 10: CHECK FOR ISSUES =====
print("\n🔍 FINAL CHECK:")
if len(classified['cat']) < 10:
    print("⚠️  Very few cat images! Consider data augmentation.")
if len(classified['dog']) < 10:
    print("⚠️  Very few dog images! Consider data augmentation.")
if len(classified['panda']) < 10:
    print("⚠️  Very few panda images! Consider data augmentation.")
if len(classified['unknown']) > 0:
    print(f"⚠️  {len(classified['unknown'])} unknown images need manual sorting in '00_UNKNOWN' folder")
else:
    print("✅ No unknown images! All images classified successfully.")

print("\n🎉 DONE! Now you can train your model using the organized dataset!")