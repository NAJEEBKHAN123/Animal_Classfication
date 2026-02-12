from torchvision import datasets
from src.data.transforms import train_transform, test_transform

# CHANGE THIS PATH - point to your existing organized dataset!
DATA_ROOT = "organized/dataset"  # NOT "data/processed"

train_dataset = datasets.ImageFolder(
    root=f'{DATA_ROOT}/train',  # Your train folder
    transform=train_transform
)

# Use val folder for validation
val_dataset = datasets.ImageFolder(
    root=f'{DATA_ROOT}/val',    # Your val folder
    transform=test_transform
)

# Use test folder for testing
test_dataset = datasets.ImageFolder(
    root=f'{DATA_ROOT}/test',   # Your test folder
    transform=test_transform
)

print(f"✅ Training classes: {train_dataset.classes}")
print(f"   Training images: {len(train_dataset)}")
print(f"   Validation images: {len(val_dataset)}")
print(f"   Test images: {len(test_dataset)}")