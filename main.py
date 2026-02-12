#!/usr/bin/env python3
"""
Animal Classifier API - Main Entry Point
Teacher's CNN with 95% accuracy
"""

import uvicorn
import argparse
import os
import sys
from pathlib import Path

def main():
    # Configure argument parser
    parser = argparse.ArgumentParser(
        description="🐾 Animal Classifier API - 95% Accuracy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py              # Run on default host:port
  python main.py --port 9000  # Run on custom port
  python main.py --host 0.0.0.0 --reload  # Run with auto-reload
        """
    )
    
    parser.add_argument(
        "--host", 
        type=str, 
        default="127.0.0.1",
        help="Host address (default: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000,
        help="Port number (default: 8000)"
    )
    
    parser.add_argument(
        "--reload", 
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="models/animal_cnn_best.pth",
        help="Path to model file (default: models/animal_cnn_best.pth)"
    )
    
    args = parser.parse_args()
    
    # Set environment variable for model path
    os.environ['MODEL_PATH'] = args.model
    
    # Create models directory if it doesn't exist
    Path("models").mkdir(exist_ok=True)
    
    # Print beautiful banner
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║     🐾  ANIMAL CLASSIFIER API - TEACHER'S CNN          ║
    ║                                                          ║
    ║         🏆 Validation: 95.18%    🎯 Test: 94.95%        ║
    ║         🐱 Cat: 93.25%    🐶 Dog: 93.60%    🐼 Panda: 97.93%  ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    print(f"📂 Model path: {args.model}")
    print(f"🚀 Server: http://{args.host}:{args.port}")
    print(f"🎨 Web UI: http://{args.host}:{args.port}")
    print(f"📚 API Docs: http://{args.host}:{args.port}/docs")
    print(f"🔍 Health: http://{args.host}:{args.port}/health")
    print(f"📊 Model Info: http://{args.host}:{args.port}/model/info")
    print("\n" + "="*60 + "\n")
    
    # Run the server
    uvicorn.run(
        "app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )

if __name__ == "__main__":
    main()