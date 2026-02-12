"""
Prediction Module for Animal CNN
Load trained model and make predictions on new images
"""

import torch
from PIL import Image
from torchvision import transforms
from src.model.cnn import CNN

class Predictor:
    def __init__(self, model_path='models/animal_cnn.pth', device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes = ['cat', 'dog', 'panda']
        
        # Load model
        self.model = CNN(num_classes=3).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.eval()
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✅ Model loaded from {model_path}")
        print(f"   Classes: {self.classes}")
        print(f"   Device: {self.device}")
    
    def predict(self, image_path):
        """Predict single image"""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
        
        class_name = self.classes[predicted.item()]
        confidence = probabilities[0][predicted.item()].item() * 100
        
        return class_name, confidence, image
    
    def predict_batch(self, image_paths):
        """Predict multiple images"""
        results = []
        for path in image_paths:
            class_name, confidence, _ = self.predict(path)
            results.append({
                'image_path': path,
                'predicted_class': class_name,
                'confidence': confidence
            })
        return results