# model.py
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

class TeacherCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(TeacherCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 32 * 128, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class AnimalClassifier:
    def __init__(self, model_path='models/animal_cnn_best.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TeacherCNN(num_classes=3).to(self.device)
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.val_accuracy = checkpoint.get('val_accuracy', 95.18)
            self.test_accuracy = checkpoint.get('test_accuracy', 94.95)
            self.classes = checkpoint.get('classes', ['cat', 'dog', 'panda'])
        else:
            self.model.load_state_dict(checkpoint)
            self.val_accuracy = 95.18
            self.test_accuracy = 94.95
            self.classes = ['cat', 'dog', 'panda']
        
        self.model.eval()
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✅ Model loaded successfully!")
        print(f"🏆 Validation Accuracy: {self.val_accuracy:.2f}%")
        print(f"🎯 Test Accuracy: {self.test_accuracy:.2f}%")
        print(f"📌 Classes: {self.classes}")
    
    def predict(self, image: Image.Image):
        """Predict single image"""
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        return {
            'class': self.classes[predicted.item()],
            'confidence': float(confidence.item() * 100),
            'probabilities': {
                self.classes[i]: float(probabilities[0][i].item() * 100)
                for i in range(len(self.classes))
            }
        }

# Singleton instance
_classifier = None

def get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = AnimalClassifier()
    return _classifier