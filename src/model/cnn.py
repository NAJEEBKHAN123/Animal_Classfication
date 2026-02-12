"""
CNN Model for Animal Classification
Teacher's exact architecture, adapted for 3 classes
Input: 128x128x3
Output: 3 classes (cat, dog, panda)
"""

from torch.nn import Module, Flatten, Sequential, Linear, ReLU
import torch
import torch.nn as nn

class CNN(Module):
    def __init__(self, num_classes=3):
        super(CNN, self).__init__()
        # Input image 128 * 128 * 3
        self.conv1d = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1) 
        self.pool1d = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # Image Size 64 * 64 * 64
        
        self.conv2d = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool2d = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # Image Size 32 * 32 * 128
        
        self.flatten_layer = Flatten(1)
        self.hidden_layers = Sequential(
            Linear(32 * 32 * 128, 256),
            ReLU(),
            Linear(256, 128),
            ReLU(),
            Linear(128, num_classes)  # Changed from 13 to 3 for animals
        )

    def forward(self, x):
        x = self.conv1d(x)
        x = torch.relu(x)
        x = self.pool1d(x)
        x = self.conv2d(x)
        x = torch.relu(x)
        x = self.pool2d(x)
        x = self.flatten_layer(x)
        x = self.hidden_layers(x)
        return x

def create_model(num_classes=3):
    """Factory function to create model"""
    return CNN(num_classes=num_classes)

if __name__ == "__main__":
    # Test model
    model = create_model(num_classes=3)
    print(f"✅ Animal CNN Model Created")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 128, 128)
    output = model(dummy_input)
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Classes: {output.shape[1]} (cat, dog, panda)")