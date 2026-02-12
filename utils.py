# utils.py
import base64
import requests
from PIL import Image
from io import BytesIO
from typing import Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def decode_base64_image(base64_string: str) -> Image.Image:
    """Decode base64 string to PIL Image"""
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        image_bytes = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        return image
    except Exception as e:
        logger.error(f"Error decoding base64 image: {e}")
        raise ValueError(f"Invalid base64 image: {str(e)}")

def load_image_from_url(url: str) -> Image.Image:
    """Load image from URL"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert('RGB')
        return image
    except Exception as e:
        logger.error(f"Error loading image from URL {url}: {e}")
        raise ValueError(f"Could not load image from URL: {str(e)}")

def process_image_input(image_input: Union[str, dict]) -> Image.Image:
    """Process either base64 string or URL from request"""
    if isinstance(image_input, dict):
        if 'image_base64' in image_input and image_input['image_base64']:
            return decode_base64_image(image_input['image_base64'])
        elif 'image_url' in image_input and image_input['image_url']:
            return load_image_from_url(image_input['image_url'])
    elif isinstance(image_input, str):
        if image_input.startswith(('http://', 'https://')):
            return load_image_from_url(image_input)
        else:
            return decode_base64_image(image_input)
    
    raise ValueError("No valid image input provided")