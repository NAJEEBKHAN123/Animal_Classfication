"""
Train-Test Split Module for Animal Dataset
Splits raw data into train/test (80/20) with proper folder structure
"""

import shutil
import random
from pathlib import Path

def split_dataset(
    source_dir="data/raw",
    target_dir="data/processed",
    train_ratio=0.8,
    val_ratio=0.0,  # Set to 0.15 if you have validation folder
    seed=42,
    image_extensions=(".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG")
):
    """
    Split dataset into train/test folders
    Expected structure:
    data/raw/
        cat/
        dog/
        panda/
    """
    random.seed(seed)
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    train_dir = target_dir / "train"
    test_dir = target_dir / "test"
    
    # Create directories
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Splitting dataset from {source_dir}")
    print(f"   Train ratio: {train_ratio*100}%")
    print(f"   Test ratio: {(1-train_ratio)*100}%")
    print("-" * 50)
    
    total_train = 0
    total_test = 0
    
    for class_dir in source_dir.iterdir():
        if not class_dir.is_dir():
            continue
        
        # Collect all images
        images = []
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in image_extensions:
                images.append(img_path)
        
        if not images:
            print(f"⚠️  No images found in {class_dir.name}")
            continue
        
        # Shuffle and split
        random.shuffle(images)
        split_idx = int(len(images) * train_ratio)
        
        train_images = images[:split_idx]
        test_images = images[split_idx:]
        
        # Create class folders
        (train_dir / class_dir.name).mkdir(exist_ok=True)
        (test_dir / class_dir.name).mkdir(exist_ok=True)
        
        # Copy images
        for img in train_images:
            shutil.copy(img, train_dir / class_dir.name / img.name)
        
        for img in test_images:
            shutil.copy(img, test_dir / class_dir.name / img.name)
        
        print(f"   {class_dir.name}: {len(train_images)} train, {len(test_images)} test")
        total_train += len(train_images)
        total_test += len(test_images)
    
    print("-" * 50)
    print(f"✅ Dataset split completed!")
    print(f"   Total train: {total_train} images")
    print(f"   Total test: {total_test} images")
    print(f"   Total: {total_train + total_test} images")

if __name__ == "__main__":
    # For animal dataset
    split_dataset(
        source_dir="data/raw",
        target_dir="data/processed",
        train_ratio=0.8,  # 80% train, 20% test
        seed=42
    )