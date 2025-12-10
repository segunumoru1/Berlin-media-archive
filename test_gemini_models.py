"""
Test which Gemini models are available
"""
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

def test_gemini_models():
    api_key = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=api_key)
    
    print("=" * 80)
    print("AVAILABLE GEMINI MODELS")
    print("=" * 80 + "\n")
    
    # List all available models
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            print(f"✅ {model.name}")
            print(f"   Description: {model.description}")
            print(f"   Methods: {', '.join(model.supported_generation_methods)}")
            print()
    
    # Test specific models
    test_models = [
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-2.0-flash-exp"
    ]
    
    print("\n" + "=" * 80)
    print("TESTING MODELS")
    print("=" * 80 + "\n")
    
    for model_name in test_models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content("Say 'Hello!'")
            print(f"✅ {model_name}: {response.text}")
        except Exception as e:
            print(f"❌ {model_name}: {e}")
    
    print("\n" + "=" * 80)
    print("RECOMMENDED: gemini-1.5-flash")
    print("=" * 80)

if __name__ == "__main__":
    test_gemini_models()