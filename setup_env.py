"""
Setup script to validate environment configuration
"""

import os
from pathlib import Path
from dotenv import load_dotenv

def setup_environment():
    """Setup and validate environment"""
    print("🔧 Setting up Berlin Media Archive environment...")
    
    # Load .env
    load_dotenv()
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_key_here":
        print("❌ OPENAI_API_KEY not set!")
        print("   Please add your OpenAI API key to .env file")
        return False
    else:
        print(f"✅ OPENAI_API_KEY configured")
    
    # Check HuggingFace token
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token or hf_token == "your_huggingface_token_here":
        print("⚠️  HUGGINGFACE_TOKEN not set (speaker diarization will be disabled)")
    else:
        print(f"✅ HUGGINGFACE_TOKEN configured")
    
    # Create directories
    directories = [
        "./data/audio",
        "./data/documents",
        "./data/vectorstore",
        "./output",
        "./logs"
    ]
    
    print("\n📁 Creating directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {directory}")
    
    print("\n✅ Environment setup complete!")
    print("\nYou can now run:")
    print("   python main.py")
    
    return True

if __name__ == "__main__":
    setup_environment()