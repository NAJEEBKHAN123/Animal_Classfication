# schemas.py
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime

class PredictionRequest(BaseModel):
    """Request model for single prediction"""
    image_base64: Optional[str] = Field(None, description="Base64 encoded image")
    image_url: Optional[str] = Field(None, description="URL of image")

class PredictionResponse(BaseModel):
    """Response model for single prediction"""
    class_name: str = Field(..., description="Predicted class (cat/dog/panda)")
    confidence: float = Field(..., description="Confidence percentage")
    probabilities: Dict[str, float] = Field(..., description="Probabilities for all classes")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: datetime = Field(default_factory=datetime.now)

class BatchPredictionRequest(BaseModel):
    """Request model for batch prediction"""
    images: List[str] = Field(..., description="List of base64 encoded images or URLs")

class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction"""
    predictions: List[PredictionResponse]
    total_time: float
    average_time: float

class ModelInfoResponse(BaseModel):
    """Model information response"""
    model_name: str = "Teacher's CNN"
    version: str = "1.0.0"
    validation_accuracy: float
    test_accuracy: float
    classes: List[str]
    input_size: int = 128
    framework: str = "PyTorch"
    device: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    device: str
    model_loaded: bool
    gpu_available: bool
    timestamp: datetime