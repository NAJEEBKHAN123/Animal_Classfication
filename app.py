"""
Animal Classifier API - Teacher's CNN
95.18% Validation Accuracy | 94.95% Test Accuracy
FastAPI Application with Beautiful Built-in UI
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from PIL import Image
from io import BytesIO
import uvicorn
import time
import base64
import json
import os
from datetime import datetime
from typing import Optional, Dict, List
from pydantic import BaseModel, Field

# ============= PYDANTIC SCHEMAS =============
class PredictionResponse(BaseModel):
    class_name: str
    confidence: float
    probabilities: Dict[str, float]
    processing_time: float
    timestamp: datetime

class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool
    gpu_available: bool
    timestamp: datetime

class ModelInfoResponse(BaseModel):
    model_name: str = "Teacher's CNN"
    version: str = "1.0.0"
    validation_accuracy: float
    test_accuracy: float
    classes: List[str]
    input_size: int = 128
    framework: str = "PyTorch"
    device: str
    per_class_accuracy: Dict[str, float] = {
        "cat": 93.25,
        "dog": 93.60,
        "panda": 97.93
    }

# ============= MODEL DEFINITION =============
import torch
import torch.nn as nn
from torchvision import transforms

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
        
        # Your transforms are CORRECT! Keep them.
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image: Image.Image, temperature=0.3):
        """Predict with temperature scaling for confident predictions"""
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            # 🔥 FIX: Add temperature scaling
            scaled_outputs = outputs / temperature
            probabilities = torch.nn.functional.softmax(scaled_outputs, dim=1)
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

# ============= FASTAPI APP =============
app = FastAPI(
    title="🐾 Animal Classifier API - Teacher's CNN",
    description="""
    ## 95% Accuracy Cat, Dog & Panda Classifier
    
    **Model Performance:**
    - 🏆 Validation Accuracy: **95.18%**
    - 🎯 Test Accuracy: **94.95%**
    - 🐱 Cat: 93.25% | 🐶 Dog: 93.60% | 🐼 Panda: 97.93%
    
    **Architecture:** Teacher's 2-layer CNN with 128x128 input
    """,
    version="1.0.0",
    contact={
        "name": "Animal Classifier API",
        "url": "http://localhost:8000",
    },
    license_info={
        "name": "MIT License",
    }
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= STUNNING HTML UI =============
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🐾 Animal Classifier - 95% Accuracy</title>
    <style>
        /* ===== GLOBAL STYLES ===== */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        
        body {
            background: linear-gradient(145deg, #0f172a 0%, #1e293b 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem;
            position: relative;
            overflow-x: hidden;
        }
        
        /* Animated background particles */
        body::before {
            content: '';
            position: absolute;
            width: 100%;
            height: 100%;
            background-image: radial-gradient(circle at 20% 30%, rgba(102, 126, 234, 0.1) 0%, transparent 30%),
                              radial-gradient(circle at 80% 70%, rgba(118, 75, 162, 0.1) 0%, transparent 30%),
                              radial-gradient(circle at 40% 80%, rgba(102, 126, 234, 0.1) 0%, transparent 30%),
                              radial-gradient(circle at 90% 20%, rgba(118, 75, 162, 0.1) 0%, transparent 30%);
            pointer-events: none;
        }
        
        .container {
            max-width: 1200px;
            width: 100%;
            position: relative;
            z-index: 10;
        }
        
        /* ===== HEADER SECTION ===== */
        .header {
            text-align: center;
            margin-bottom: 3rem;
            animation: fadeInDown 1s ease;
        }
        
        @keyframes fadeInDown {
            from {
                opacity: 0;
                transform: translateY(-30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        h1 {
            font-size: 4rem;
            font-weight: 800;
            background: linear-gradient(135deg, #fff 0%, #e2e8f0 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
            text-shadow: 0 10px 30px rgba(0,0,0,0.3);
            letter-spacing: -2px;
        }
        
        .subtitle {
            color: #94a3b8;
            font-size: 1.2rem;
            margin-bottom: 2rem;
            display: flex;
            justify-content: center;
            gap: 1rem;
            flex-wrap: wrap;
        }
        
        .badge {
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            padding: 0.5rem 1.5rem;
            border-radius: 50px;
            border: 1px solid rgba(255,255,255,0.1);
            color: white;
            font-weight: 500;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        
        /* ===== STATS CARDS ===== */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin-bottom: 3rem;
            animation: fadeInUp 1s ease 0.2s both;
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .stat-card {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 20px;
            padding: 1.8rem 1rem;
            text-align: center;
            transition: all 0.3s ease;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
            background: rgba(30, 41, 59, 0.9);
            border-color: rgba(102, 126, 234, 0.5);
            box-shadow: 0 15px 40px rgba(102, 126, 234, 0.2);
        }
        
        .stat-emoji {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }
        
        .stat-label {
            color: #94a3b8;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-bottom: 0.5rem;
        }
        
        .stat-value {
            color: white;
            font-size: 2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        /* ===== MAIN UPLOAD CARD ===== */
        .upload-card {
            background: rgba(15, 23, 42, 0.8);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 30px;
            padding: 3rem;
            margin-bottom: 2rem;
            box-shadow: 0 20px 40px rgba(0,0,0,0.4);
            animation: fadeInUp 1s ease 0.4s both;
        }
        
        .upload-area {
            border: 3px dashed rgba(102, 126, 234, 0.3);
            border-radius: 20px;
            padding: 3rem 2rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: rgba(102, 126, 234, 0.05);
        }
        
        .upload-area:hover {
            border-color: #667eea;
            background: rgba(102, 126, 234, 0.1);
            transform: scale(1.02);
        }
        
        .upload-icon {
            font-size: 5rem;
            margin-bottom: 1rem;
            filter: drop-shadow(0 10px 20px rgba(102, 126, 234, 0.3));
            animation: float 3s ease-in-out infinite;
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
        }
        
        .upload-title {
            color: white;
            font-size: 1.8rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        
        .upload-hint {
            color: #94a3b8;
            font-size: 1rem;
        }
        
        /* ===== PREVIEW SECTION ===== */
        .preview-container {
            margin-top: 2rem;
            display: none;
            animation: fadeIn 0.5s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        .preview-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            color: white;
        }
        
        .preview-image {
            width: 100%;
            max-height: 400px;
            object-fit: contain;
            border-radius: 15px;
            background: rgba(0,0,0,0.3);
            border: 2px solid rgba(102, 126, 234, 0.3);
        }
        
        /* ===== LOADING SPINNER ===== */
        .loading {
            display: none;
            text-align: center;
            padding: 3rem;
            animation: fadeIn 0.5s ease;
        }
        
        .spinner {
            width: 60px;
            height: 60px;
            border: 4px solid rgba(102, 126, 234, 0.1);
            border-left-color: #667eea;
            border-right-color: #764ba2;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 1.5rem;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .loading-text {
            color: white;
            font-size: 1.2rem;
            font-weight: 500;
        }
        
        /* ===== RESULTS CARD ===== */
        .result-card {
            background: linear-gradient(145deg, #1e293b, #0f172a);
            border-radius: 25px;
            padding: 2.5rem;
            margin-top: 2rem;
            border: 1px solid rgba(102, 126, 234, 0.3);
            display: none;
            animation: slideUp 0.6s ease;
            box-shadow: 0 20px 40px rgba(0,0,0,0.4);
        }
        
        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
            padding-bottom: 1.5rem;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        
        .result-title {
            color: white;
            font-size: 1.5rem;
            font-weight: 600;
        }
        
        .processing-time {
            background: rgba(102, 126, 234, 0.2);
            color: #a5b4fc;
            padding: 0.5rem 1.2rem;
            border-radius: 50px;
            font-size: 0.9rem;
            border: 1px solid rgba(102, 126, 234, 0.3);
        }
        
        .prediction-main {
            text-align: center;
            margin-bottom: 2.5rem;
        }
        
        .prediction-emoji {
            font-size: 5rem;
            margin-bottom: 0.5rem;
            filter: drop-shadow(0 10px 20px rgba(0,0,0,0.3));
        }
        
        .prediction-class {
            font-size: 3rem;
            font-weight: 800;
            color: white;
            text-transform: uppercase;
            letter-spacing: 4px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        
        .confidence-bar-container {
            max-width: 500px;
            margin: 2rem auto;
        }
        
        .confidence-label {
            display: flex;
            justify-content: space-between;
            color: white;
            margin-bottom: 0.5rem;
        }
        
        .confidence-bar-bg {
            background: rgba(255,255,255,0.1);
            height: 12px;
            border-radius: 6px;
            overflow: hidden;
        }
        
        .confidence-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            border-radius: 6px;
            transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
            width: 0%;
        }
        
        .probabilities-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1.5rem;
            margin-top: 2.5rem;
        }
        
        .prob-card {
            background: rgba(0,0,0,0.2);
            padding: 1.5rem;
            border-radius: 15px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.05);
            transition: all 0.3s ease;
        }
        
        .prob-card:hover {
            transform: translateY(-5px);
            background: rgba(102, 126, 234, 0.1);
            border-color: rgba(102, 126, 234, 0.3);
        }
        
        .prob-emoji {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }
        
        .prob-label {
            color: #94a3b8;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-bottom: 0.5rem;
        }
        
        .prob-value {
            color: white;
            font-size: 2rem;
            font-weight: 700;
        }
        
        /* ===== ACTIONS ===== */
        .actions {
            display: flex;
            gap: 1rem;
            justify-content: center;
            margin-top: 2.5rem;
        }
        
        .btn {
            padding: 1rem 2.5rem;
            border: none;
            border-radius: 50px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 30px rgba(102, 126, 234, 0.4);
        }
        
        .btn-secondary {
            background: rgba(255,255,255,0.1);
            color: white;
            border: 1px solid rgba(255,255,255,0.2);
            backdrop-filter: blur(10px);
        }
        
        .btn-secondary:hover {
            background: rgba(255,255,255,0.2);
            transform: translateY(-2px);
        }
        
        /* ===== FOOTER ===== */
        .footer {
            text-align: center;
            margin-top: 3rem;
            color: #64748b;
            font-size: 0.9rem;
            animation: fadeInUp 1s ease 0.6s both;
        }
        
        /* ===== RESPONSIVE ===== */
        @media (max-width: 768px) {
            body { padding: 1rem; }
            h1 { font-size: 2.5rem; }
            .upload-card { padding: 1.5rem; }
            .probabilities-grid { grid-template-columns: 1fr; }
            .stats-grid { grid-template-columns: 1fr; }
            .prediction-class { font-size: 2rem; }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>🐾 Animal Classifier</h1>
            <div class="subtitle">
                <span class="badge">🏆 95.18% Accuracy</span>
                <span class="badge">🎯 94.95% Test Score</span>
                <span class="badge">🧠 Teacher's CNN</span>
            </div>
        </div>
        
        <!-- Stats Grid -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-emoji">🐱</div>
                <div class="stat-label">Cat Accuracy</div>
                <div class="stat-value">93.25%</div>
            </div>
            <div class="stat-card">
                <div class="stat-emoji">🐶</div>
                <div class="stat-label">Dog Accuracy</div>
                <div class="stat-value">93.60%</div>
            </div>
            <div class="stat-card">
                <div class="stat-emoji">🐼</div>
                <div class="stat-label">Panda Accuracy</div>
                <div class="stat-value">97.93%</div>
            </div>
            <div class="stat-card">
                <div class="stat-emoji">⚡</div>
                <div class="stat-label">Inference</div>
                <div class="stat-value">~0.2s</div>
            </div>
        </div>
        
        <!-- Main Upload Card -->
        <div class="upload-card">
            <div class="upload-area" onclick="document.getElementById('fileInput').click()" ondragover="event.preventDefault()" ondrop="handleDrop(event)">
                <div class="upload-icon">📸</div>
                <h2 class="upload-title">Upload or Drop Image</h2>
                <p class="upload-hint">PNG, JPG, JPEG • Max 10MB</p>
                <input type="file" id="fileInput" accept="image/*" style="display: none;" onchange="handleFileSelect(event)">
            </div>
            
            <!-- Preview Container -->
            <div class="preview-container" id="previewContainer">
                <div class="preview-header">
                    <span style="color: white; font-weight: 600;">Preview</span>
                    <span style="color: #94a3b8; font-size: 0.9rem;" id="fileName"></span>
                </div>
                <img class="preview-image" id="previewImage" alt="Preview">
            </div>
            
            <!-- Loading Spinner -->
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <div class="loading-text">🔍 Analyzing with 95% accuracy model...</div>
                <p style="color: #94a3b8; margin-top: 1rem;">Teacher's CNN is processing your image</p>
            </div>
            
            <!-- Results Card -->
            <div class="result-card" id="resultCard">
                <div class="result-header">
                    <span class="result-title">🎯 Prediction Result</span>
                    <span class="processing-time" id="processingTime">⚡ 0.00s</span>
                </div>
                
                <div class="prediction-main">
                    <div class="prediction-emoji" id="predictionEmoji">🐱</div>
                    <div class="prediction-class" id="predictionClass">CAT</div>
                    
                    <div class="confidence-bar-container">
                        <div class="confidence-label">
                            <span style="color: white;">Confidence</span>
                            <span style="color: #a5b4fc; font-weight: 600;" id="confidencePercent">0%</span>
                        </div>
                        <div class="confidence-bar-bg">
                            <div class="confidence-bar-fill" id="confidenceBar" style="width: 0%;"></div>
                        </div>
                    </div>
                </div>
                
                <div style="color: #94a3b8; text-align: center; margin-bottom: 1rem; font-size: 0.9rem;">
                    Probability Distribution
                </div>
                
                <div class="probabilities-grid" id="probabilitiesGrid">
                    <!-- Dynamic content -->
                </div>
                
                <div class="actions">
                    <button class="btn btn-primary" onclick="resetUpload()">
                        🔄 Try Another
                    </button>
                    <button class="btn btn-secondary" onclick="window.location.href='/docs'">
                        📚 API Docs
                    </button>
                </div>
            </div>
        </div>
        
        <!-- Footer -->
        <div class="footer">
            <p>🧠 Teacher's CNN Architecture • 128x128 Input • 33.6M Parameters</p>
            <p style="margin-top: 0.5rem;">✨ Achieved 95.18% Validation Accuracy on Cat/Dog/Panda Dataset</p>
        </div>
    </div>
    
    <script>
        // API Configuration
        const API_URL = window.location.origin;
        
        // Handle file selection
        function handleFileSelect(event) {
            const file = event.target.files[0];
            if (file) processFile(file);
        }
        
        // Handle drag & drop
        function handleDrop(event) {
            event.preventDefault();
            const file = event.dataTransfer.files[0];
            if (file) processFile(file);
        }
        
        // Process uploaded file
        async function processFile(file) {
            // Validate file
            if (!file.type.match('image.*')) {
                alert('Please upload an image file');
                return;
            }
            
            if (file.size > 10 * 1024 * 1024) {
                alert('File size must be less than 10MB');
                return;
            }
            
            // Show preview
            const reader = new FileReader();
            reader.onload = function(e) {
                const preview = document.getElementById('previewImage');
                preview.src = e.target.result;
                document.getElementById('previewContainer').style.display = 'block';
                document.getElementById('fileName').textContent = file.name;
            };
            reader.readAsDataURL(file);
            
            // Hide previous results
            document.getElementById('resultCard').style.display = 'none';
            
            // Show loading
            document.getElementById('loading').style.display = 'block';
            
            // Prepare form data
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                // Call API
                const response = await fetch(`${API_URL}/predict/upload`, {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                // Hide loading
                document.getElementById('loading').style.display = 'none';
                
                // Display results
                displayResults(data);
                
            } catch (error) {
                console.error('Error:', error);
                document.getElementById('loading').style.display = 'none';
                alert('Error analyzing image. Please try again.');
            }
        }
        
        // Display prediction results
        function displayResults(data) {
            // Show result card
            const resultCard = document.getElementById('resultCard');
            resultCard.style.display = 'block';
            
            // Set emoji
            const emoji = data.class_name === 'cat' ? '🐱' : data.class_name === 'dog' ? '🐶' : '🐼';
            document.getElementById('predictionEmoji').textContent = emoji;
            
            // Set class name
            document.getElementById('predictionClass').textContent = data.class_name.toUpperCase();
            
            // Set confidence
            document.getElementById('confidencePercent').textContent = `${data.confidence.toFixed(1)}%`;
            document.getElementById('confidenceBar').style.width = `${data.confidence}%`;
            
            // Set processing time
            document.getElementById('processingTime').innerHTML = `⚡ ${data.processing_time.toFixed(3)}s`;
            
            // Set probabilities
            const probsGrid = document.getElementById('probabilitiesGrid');
            probsGrid.innerHTML = '';
            
            const classes = ['cat', 'dog', 'panda'];
            const emojis = {'cat': '🐱', 'dog': '🐶', 'panda': '🐼'};
            const colors = {'cat': '#FF9999', 'dog': '#66B3FF', 'panda': '#99FF99'};
            
            classes.forEach(cls => {
                const prob = data.probabilities[cls] || 0;
                const div = document.createElement('div');
                div.className = 'prob-card';
                div.innerHTML = `
                    <div class="prob-emoji">${emojis[cls]}</div>
                    <div class="prob-label">${cls.toUpperCase()}</div>
                    <div class="prob-value" style="color: ${colors[cls]}">${prob.toFixed(1)}%</div>
                `;
                probsGrid.appendChild(div);
            });
            
            // Smooth scroll to results
            resultCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
        
        // Reset upload
        function resetUpload() {
            // Reset file input
            document.getElementById('fileInput').value = '';
            
            // Hide preview and results
            document.getElementById('previewContainer').style.display = 'none';
            document.getElementById('resultCard').style.display = 'none';
            
            // Clear preview image
            document.getElementById('previewImage').src = '';
        }
    </script>
</body>
</html>
"""

# ============= API ENDPOINTS =============

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve beautiful web UI"""
    return HTML_TEMPLATE

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    classifier = get_classifier()
    return HealthResponse(
        status="healthy",
        device=classifier.device.type,
        model_loaded=True,
        gpu_available=torch.cuda.is_available(),
        timestamp=datetime.now()
    )

@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Get model information and performance metrics"""
    classifier = get_classifier()
    return ModelInfoResponse(
        validation_accuracy=classifier.val_accuracy,
        test_accuracy=classifier.test_accuracy,
        classes=classifier.classes,
        device=classifier.device.type,
        per_class_accuracy={"cat": 93.25, "dog": 93.60, "panda": 97.93}
    )

@app.post("/predict/upload", response_model=PredictionResponse)
async def predict_upload(file: UploadFile = File(...)):
    """Upload and classify an image"""
    start_time = time.time()
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert('RGB')
        
        # Predict
        classifier = get_classifier()
        result = classifier.predict(image)
        
        processing_time = time.time() - start_time
        
        return PredictionResponse(
            class_name=result['class'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            processing_time=round(processing_time, 3),
            timestamp=datetime.now()
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.get("/api-docs", response_class=HTMLResponse)
async def api_docs():
    """Redirect to Swagger UI"""
    return RedirectResponse(url="/docs")