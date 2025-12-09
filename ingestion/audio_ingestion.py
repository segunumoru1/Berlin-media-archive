"""
Audio Ingestion Pipeline
Handles audio transcription and speaker diarization with Whisper.
"""

import os
import subprocess
import shutil
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass, asdict
import whisper
import torch
from loguru import logger


def check_ffmpeg():
    """Check if ffmpeg is available."""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        logger.info(f"✅ FFmpeg found at: {ffmpeg_path}")
        return True
    else:
        logger.error("❌ FFmpeg not found in PATH")
        logger.error("   Please install FFmpeg:")
        logger.error("   - Windows: choco install ffmpeg  OR  download from https://www.gyan.dev/ffmpeg/builds/")
        logger.error("   - Mac: brew install ffmpeg")
        logger.error("   - Linux: sudo apt install ffmpeg")
        return False


@dataclass
class TranscriptSegment:
    """Represents a segment of transcribed audio."""
    text: str
    start_time: float
    end_time: float
    speaker: Optional[str] = None
    confidence: Optional[float] = None
    
    def to_dict(self):
        return asdict(self)


class AudioIngestionPipeline:
    """
    Production-grade audio ingestion pipeline.
    Handles transcription with Whisper and optional speaker diarization.
    """
    
    def __init__(
        self,
        model_size: str = None,
        enable_diarization: bool = None,
        chunk_length: int = None,
        device: Optional[str] = None
    ):
        """
        Initialize audio ingestion pipeline.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            enable_diarization: Enable speaker diarization
            chunk_length: Length of audio chunks in seconds
            device: Device to use (cuda/cpu)
        """
        # Check FFmpeg first
        if not check_ffmpeg():
            raise RuntimeError("FFmpeg is not installed. Please install FFmpeg to process audio files.")
        
        self.model_size = model_size or os.getenv("WHISPER_MODEL", "base")
        self.enable_diarization = enable_diarization if enable_diarization is not None else os.getenv("ENABLE_DIARIZATION", "false").lower() == "true"
        self.chunk_length = chunk_length or int(os.getenv("AUDIO_CHUNK_LENGTH", "30"))
        
        # Determine device
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading Whisper model: {self.model_size}")
        try:
            self.whisper_model = whisper.load_model(self.model_size, device=self.device)
            logger.info(f"✅ Whisper model loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
        
        # Speaker diarization
        self.diarization_pipeline = None
        if self.enable_diarization:
            hf_token = os.getenv("HUGGINGFACE_TOKEN")
            if hf_token and hf_token != "your_huggingface_token_here":
                try:
                    from pyannote.audio import Pipeline
                    
                    logger.info("Loading speaker diarization pipeline...")
                    self.diarization_pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1",
                        use_auth_token=hf_token
                    )
                    
                    if torch.cuda.is_available():
                        self.diarization_pipeline.to(torch.device("cuda"))
                    
                    logger.info("✅ Speaker diarization enabled")
                except Exception as e:
                    logger.warning(f"Failed to load diarization pipeline: {e}")
                    logger.warning("Speaker diarization disabled")
                    self.enable_diarization = False
            else:
                logger.warning("HUGGINGFACE_TOKEN not set, diarization disabled")
                self.enable_diarization = False
        
        logger.info(f"Audio ingestion pipeline initialized (model={self.model_size}, device={self.device}, diarization={self.enable_diarization})")
    
    def ingest_audio(
        self,
        audio_path: str,
        output_dir: Optional[str] = None
    ) -> List[TranscriptSegment]:
        """
        Ingest an audio file.
        
        Args:
            audio_path: Path to audio file
            output_dir: Directory to save processed output
            
        Returns:
            List of TranscriptSegment objects
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        logger.info(f"Ingesting audio: {audio_path.name}")
        logger.info(f"File size: {audio_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        try:
            # Transcribe audio
            transcription = self.transcribe_audio(str(audio_path))
            
            if not transcription:
                raise ValueError("Transcription failed - no segments returned")
            
            # Apply speaker diarization if enabled
            if self.enable_diarization and self.diarization_pipeline:
                try:
                    segments = self.apply_diarization(str(audio_path), transcription)
                    logger.info(f"✅ Diarization applied, identified {len(set(s.speaker for s in segments if s.speaker))} speakers")
                except Exception as e:
                    logger.warning(f"Diarization failed: {e}, using transcription without speakers")
                    segments = transcription
            else:
                segments = transcription
            
            # Save output if directory specified
            if output_dir:
                self.save_segments(segments, audio_path.name, output_dir)
            
            logger.info(f"✅ Audio ingestion complete: {len(segments)} segments")
            
            return segments
            
        except Exception as e:
            logger.error(f"Audio ingestion failed: {e}", exc_info=True)
            raise
    
    def transcribe_audio(self, audio_path: str) -> List[TranscriptSegment]:
        """
        Transcribe audio using Whisper.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of TranscriptSegment objects
        """
        logger.info(f"Transcribing: {audio_path}")
        
        # Verify file exists
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Verify FFmpeg can read the file
        try:
            result = subprocess.run(
                ["ffmpeg", "-i", audio_path, "-f", "null", "-"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                logger.warning(f"FFmpeg pre-check warning: {result.stderr[:200]}")
        except FileNotFoundError:
            raise RuntimeError("FFmpeg not found. Please install FFmpeg.")
        except Exception as e:
            logger.warning(f"FFmpeg pre-check failed: {e}")
        
        try:
            # Transcribe with Whisper
            result = self.whisper_model.transcribe(
                audio_path,
                language="en",
                task="transcribe",
                verbose=False
            )
            
            if not result or "segments" not in result:
                raise ValueError("Whisper transcription returned no segments")
            
            # Convert to TranscriptSegment objects
            segments = []
            for seg in result["segments"]:
                segment = TranscriptSegment(
                    text=seg["text"].strip(),
                    start_time=seg["start"],
                    end_time=seg["end"],
                    speaker=None,  # Will be filled by diarization if enabled
                    confidence=seg.get("no_speech_prob")
                )
                segments.append(segment)
            
            logger.info(f"✅ Transcribed {len(segments)} segments from audio")
            
            return segments
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}", exc_info=True)
            raise
    
    def apply_diarization(
        self,
        audio_path: str,
        transcription: List[TranscriptSegment]
    ) -> List[TranscriptSegment]:
        """
        Apply speaker diarization to transcription segments.
        
        Args:
            audio_path: Path to audio file
            transcription: List of transcription segments
            
        Returns:
            Segments with speaker labels
        """
        if not self.diarization_pipeline:
            logger.warning("Diarization pipeline not available")
            return transcription
        
        logger.info("Applying speaker diarization...")
        
        try:
            # Run diarization
            diarization = self.diarization_pipeline(audio_path)
            
            # Map speakers to segments
            for segment in transcription:
                segment_start = segment.start_time
                segment_end = segment.end_time
                
                # Find overlapping speaker
                max_overlap = 0
                best_speaker = None
                
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    overlap_start = max(segment_start, turn.start)
                    overlap_end = min(segment_end, turn.end)
                    overlap = max(0, overlap_end - overlap_start)
                    
                    if overlap > max_overlap:
                        max_overlap = overlap
                        best_speaker = speaker
                
                segment.speaker = best_speaker if best_speaker else "UNKNOWN"
            
            # Count unique speakers
            speakers = set(s.speaker for s in transcription if s.speaker)
            logger.info(f"Identified {len(speakers)} unique speakers: {speakers}")
            
            return transcription
            
        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            return transcription
    
    def save_segments(
        self,
        segments: List[TranscriptSegment],
        filename: str,
        output_dir: str
    ):
        """
        Save transcription segments to JSON file.
        
        Args:
            segments: List of segments
            filename: Original filename
            output_dir: Output directory
        """
        import json
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{Path(filename).stem}_transcript.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump([seg.to_dict() for seg in segments], f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved transcript to: {output_file}")