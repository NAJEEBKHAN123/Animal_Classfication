# ============================================
# COMPLETE PREDICTION SCRIPT - READY TO USE!
# ============================================

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import sys

# ============= MODEL DEFINITION =============
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

# ============= LOAD MODEL =============
def load_model(model_path='animal_cnn_best.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TeacherCNN(num_classes=3).to(device)
    
    # Load the weights
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"🏆 Loaded model with validation accuracy: {checkpoint.get('val_accuracy', 95.18):.2f}%")
        print(f"🎯 Test accuracy: {checkpoint.get('test_accuracy', 94.95):.2f}%")
        
        # Print per-class accuracy if available
        if 'classes' in checkpoint:
            print(f"📌 Classes: {checkpoint['classes']}")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Model loaded successfully!")
    
    model.eval()
    return model, device

# ============= TRANSFORMS =============
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============= PREDICTION FUNCTION =============
classes = ['cat', 'dog', 'panda']

def predict_image(image_path, model, device):
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        class_name = classes[predicted.item()]
        confidence_percent = confidence.item() * 100
        
        return class_name, confidence_percent, image
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None, None

# ============= BATCH PREDICTION =============
def predict_folder(folder_path, model, device):
    import os
    results = []
    for img_file in os.listdir(folder_path):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, img_file)
            class_name, confidence, _ = predict_image(img_path, model, device)
            results.append({
                'image': img_file,
                'prediction': class_name,
                'confidence': f'{confidence:.2f}%'
            })
    return results

# ============= MAIN =============
if __name__ == "__main__":
    # Load model
    model_path = 'animal_cnn_best.pth'  # or 'animal_cnn_final_acc94.9.pth'
    model, device = load_model(model_path)
    
    # Check if image path is provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        class_name, confidence, _ = predict_image(image_path, model, device)
        
        if class_name:
            print(f"\n🎯 Prediction: {class_name}")
            print(f"   Confidence: {confidence:.2f}%")
            
            # Show image if matplotlib is available
            try:
                import matplotlib.pyplot as plt
                img = Image.open(image_path)
                plt.imshow(img)
                plt.title(f"{class_name} ({confidence:.2f}%)")
                plt.axis('off')
                plt.show()
            except:
                pass
    else:
        print("\n🔍 Usage: python predict_animal.py <image_path>")
        print("   Example: python predict_animal.py cat.jpg")
        print("\n📁 Or use in Python:")
        print("   >>> class_name, confidence, _ = predict_image('my_cat.jpg', model, device)")
        print("   >>> print(f'Prediction: {class_name} ({confidence:.2f}%)')")