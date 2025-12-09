"""
Check if FFmpeg is installed
"""
import shutil
import subprocess

def check_ffmpeg():
    print("=" * 60)
    print("FFMPEG CHECK")
    print("=" * 60)
    
    # Check if ffmpeg is in PATH
    ffmpeg_path = shutil.which("ffmpeg")
    
    if ffmpeg_path:
        print(f"✅ FFmpeg found at: {ffmpeg_path}")
        
        # Get version
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            version_line = result.stdout.split('\n')[0]
            print(f"✅ Version: {version_line}")
            print("\n✅ FFmpeg is installed and working!")
            
        except Exception as e:
            print(f"⚠️  FFmpeg found but can't get version: {e}")
    
    else:
        print("❌ FFmpeg NOT found in PATH")
        print("\nTo install FFmpeg:")
        print("\n  Option 1 - Chocolatey (recommended):")
        print("    choco install ffmpeg")
        print("\n  Option 2 - Manual:")
        print("    1. Download from: https://www.gyan.dev/ffmpeg/builds/")
        print("    2. Extract to: C:\\ffmpeg")
        print("    3. Add to PATH: C:\\ffmpeg\\bin")
        print("    4. Restart terminal")
        
        return False
    
    print("=" * 60)
    return True

if __name__ == "__main__":
    check_ffmpeg()