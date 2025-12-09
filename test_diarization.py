"""
Test speaker diarization setup
"""
import os
from dotenv import load_dotenv

load_dotenv()

def test_diarization():
    print("=" * 60)
    print("SPEAKER DIARIZATION TEST")
    print("=" * 60)
    
    # Check HuggingFace token
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    
    if not hf_token or hf_token == "your_huggingface_token_here":
        print("\n❌ HUGGINGFACE_TOKEN not set")
        print("\nTo enable speaker diarization:")
        print("1. Go to: https://huggingface.co/settings/tokens")
        print("2. Create a new token with read access")
        print("3. Accept terms at: https://huggingface.co/pyannote/speaker-diarization-3.1")
        print("4. Add to .env: HUGGINGFACE_TOKEN=hf_xxxxx")
        return False
    
    print(f"\n✅ HuggingFace token found: {hf_token[:10]}...{hf_token[-4:]}")
    
    # Try to load pipeline
    try:
        print("\n📦 Loading pyannote.audio...")
        from pyannote.audio import Pipeline
        
        print("✅ pyannote.audio imported successfully")
        
        print("\n📥 Loading diarization pipeline...")
        print("   (This may take a minute on first run...)")
        
        # Try new parameter name
        try:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=hf_token
            )
            print("✅ Pipeline loaded successfully (using 'token' parameter)")
        except TypeError:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )
            print("✅ Pipeline loaded successfully (using 'use_auth_token' parameter)")
        
        print("\n🎉 Speaker diarization is ready!")
        print("   You can now upload audio with enable_diarization=true")
        
        return True
        
    except ImportError as e:
        print(f"\n❌ pyannote.audio not installed")
        print(f"   Install with: pip install pyannote.audio==3.1.1")
        return False
    except Exception as e:
        print(f"\n❌ Failed to load diarization pipeline: {e}")
        print("\nTroubleshooting:")
        print("1. Check your HuggingFace token is valid")
        print("2. Accept terms at: https://huggingface.co/pyannote/speaker-diarization-3.1")
        print("3. Try: pip install --upgrade pyannote.audio")
        return False

if __name__ == "__main__":
    test_diarization()