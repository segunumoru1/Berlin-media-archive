"""
Test speaker diarization setup
"""
import os
from dotenv import load_dotenv

load_dotenv()

def test_diarization_setup():
    print("=" * 80)
    print("SPEAKER DIARIZATION SETUP TEST")
    print("=" * 80)
    
    # 1. Check HuggingFace token
    print("\n1. Checking HuggingFace Token...")
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    
    if not hf_token or hf_token == "your_huggingface_token_here":
        print("   ❌ HUGGINGFACE_TOKEN not set in .env")
        print("\n   To enable speaker diarization:")
        print("   1. Create account at https://huggingface.co")
        print("   2. Get token from https://huggingface.co/settings/tokens")
        print("   3. Accept license at https://huggingface.co/pyannote/speaker-diarization-3.1")
        print("   4. Add to .env: HUGGINGFACE_TOKEN=hf_...")
        return False
    else:
        print(f"   ✅ Token found: {hf_token[:10]}...")
    
    # 2. Check pyannote.audio
    print("\n2. Checking pyannote.audio...")
    try:
        import pyannote.audio
        print(f"   ✅ pyannote.audio version: {pyannote.audio.__version__}")
    except ImportError as e:
        print(f"   ❌ pyannote.audio not installed: {e}")
        print("\n   Install with:")
        print("   pip install pyannote.audio==3.1.1 pyannote.core==5.0.0 speechbrain==0.5.16")
        return False
    
    # 3. Check dependencies
    print("\n3. Checking dependencies...")
    deps = {
        "torch": None,
        "torchaudio": None,
        "speechbrain": None,
        "pyannote.core": None
    }
    
    for dep in deps:
        try:
            if dep == "pyannote.core":
                import pyannote.core
                deps[dep] = pyannote.core.__version__
            elif dep == "speechbrain":
                import speechbrain
                deps[dep] = speechbrain.__version__
            else:
                mod = __import__(dep)
                deps[dep] = mod.__version__
            print(f"   ✅ {dep}: {deps[dep]}")
        except ImportError:
            print(f"   ❌ {dep}: Not installed")
            return False
    
    # 4. Test loading pipeline
    print("\n4. Testing diarization pipeline...")
    try:
        from pyannote.audio import Pipeline
        
        print("   Loading model from HuggingFace...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        print("   ✅ Diarization pipeline loaded successfully!")
        
        # Check device
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Device: {device}")
        
        if device == "cuda":
            pipeline.to(torch.device("cuda"))
            print("   ✅ Moved pipeline to GPU")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Failed to load pipeline: {e}")
        print("\n   Common issues:")
        print("   - Invalid HuggingFace token")
        print("   - License not accepted at https://huggingface.co/pyannote/speaker-diarization-3.1")
        print("   - Missing dependencies")
        return False
    
    print("\n" + "=" * 80)
    print("✅ Speaker diarization is ready!")
    print("=" * 80)


if __name__ == "__main__":
    success = test_diarization_setup()
    
    if not success:
        print("\n❌ Diarization setup incomplete")
        print("Audio ingestion will work but speaker labels won't be added")
    else:
        print("\n✅ All checks passed! Speaker diarization is enabled.")