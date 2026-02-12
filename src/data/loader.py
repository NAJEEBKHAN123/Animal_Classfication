"""
DataLoader Module for Animal CNN
Creates optimized data loaders for training and testing
"""

import torch
from torch.utils.data import DataLoader
from src.data.dataset import train_dataset, test_dataset, val_dataset

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Generator for CUDA compatibility
generator = torch.Generator(device=DEVICE)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,  # Increased from 16 for faster training
    shuffle=True,
    num_workers=2,
    pin_memory=True if DEVICE == 'cuda' else False,
    generator=generator
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=True if DEVICE == 'cuda' else False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=True if DEVICE == 'cuda' else False
)

print(f"✅ DataLoaders created:")
print(f"   Train batches: {len(train_loader)}")
print(f"   Val batches: {len(val_loader)}")
print(f"   Test batches: {len(test_loader)}")