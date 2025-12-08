"""
Audio Ingestion Pipeline
Handles audio transcription with timestamp preservation and speaker diarization.
"""

import os
import json
import whisper
import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from pydub import AudioSegment
from loguru import logger

from utils.config import settings


@dataclass
class TranscriptSegment:
    """Represents a single transcript segment with metadata."""
    text: str
    start_time: float
    end_time: float
    speaker: Optional[str] = None
    confidence: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def get_timestamp_str(self) -> str:
        """Get formatted timestamp string."""
        return f"{self._format_time(self.start_time)} - {self._format_time(self.end_time)}"
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds to MM:SS or HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"


class AudioIngestionPipeline:
    """Pipeline for ingesting and transcribing audio files."""
    
    def __init__(
        self,
        whisper_model: str = None,
        enable_diarization: bool = True
    ):
        """
        Initialize audio ingestion pipeline.
        
        Args:
            whisper_model: Whisper model size (tiny, base, small, medium, large)
            enable_diarization: Whether to enable speaker diarization
        """
        self.whisper_model_name = whisper_model or settings.whisper_model
        self.enable_diarization = enable_diarization and settings.enable_speaker_diarization
        
        logger.info(f"Initializing AudioIngestionPipeline with model: {self.whisper_model_name}")
        
        # Load Whisper model
        try:
            self.whisper_model = whisper.load_model(self.whisper_model_name)
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
        
        # Initialize diarization pipeline if enabled
        self.diarization_pipeline = None
        if self.enable_diarization:
            try:
                self._initialize_diarization()
            except Exception as e:
                logger.warning(f"Failed to initialize diarization: {e}. Continuing without diarization.")
                self.enable_diarization = False
    
    def _initialize_diarization(self):
        """Initialize speaker diarization pipeline."""
        try:
            from pyannote.audio import Pipeline
            
            # Check if HuggingFace token is available
            if not settings.huggingface_token:
                logger.warning("HUGGINGFACE_TOKEN not set. Diarization disabled.")
                return
            
            # Load pretrained pipeline
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=settings.huggingface_token
            )
            
            # Use GPU if available
            if torch.cuda.is_available():
                self.diarization_pipeline.to(torch.device("cuda"))
                logger.info("Diarization pipeline loaded on GPU")
            else:
                logger.info("Diarization pipeline loaded on CPU")
                
        except Exception as e:
            logger.error(f"Failed to initialize diarization: {e}")
            raise
    
    def ingest_audio(
        self,
        audio_path: str,
        output_dir: Optional[str] = None
    ) -> Tuple[List[TranscriptSegment], Dict]:
        """
        Ingest and transcribe audio file with timestamps.
        
        Args:
            audio_path: Path to audio file
            output_dir: Optional directory to save transcription results
            
        Returns:
            Tuple of (transcript_segments, metadata)
        """
        try:
            audio_path = Path(audio_path)
            logger.info(f"Ingesting audio file: {audio_path}")
            
            # Validate file exists
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Get audio metadata
            metadata = self._get_audio_metadata(audio_path)
            logger.info(f"Audio metadata: duration={metadata['duration_seconds']:.2f}s")
            
            # Transcribe with Whisper
            logger.info("Starting transcription...")
            transcription_result = self._transcribe_with_whisper(audio_path)
            
            # Extract segments with timestamps
            segments = self._extract_segments(transcription_result)
            logger.info(f"Extracted {len(segments)} transcript segments")
            
            # Perform speaker diarization if enabled
            if self.enable_diarization and self.diarization_pipeline:
                logger.info("Performing speaker diarization...")
                segments = self._apply_diarization(audio_path, segments)
            
            # Save results if output directory specified
            if output_dir:
                self._save_results(audio_path, segments, metadata, output_dir)
            
            logger.info("Audio ingestion completed successfully")
            return segments, metadata
            
        except Exception as e:
            logger.error(f"Audio ingestion failed: {e}", exc_info=True)
            raise
    
    def _get_audio_metadata(self, audio_path: Path) -> Dict:
        """Extract audio file metadata."""
        try:
            audio = AudioSegment.from_file(str(audio_path))
            
            return {
                "filename": audio_path.name,
                "filepath": str(audio_path),
                "duration_seconds": len(audio) / 1000.0,
                "channels": audio.channels,
                "frame_rate": audio.frame_rate,
                "sample_width": audio.sample_width,
                "file_size_mb": audio_path.stat().st_size / (1024 * 1024)
            }
        except Exception as e:
            logger.warning(f"Failed to extract metadata with pydub: {e}")
            return {
                "filename": audio_path.name,
                "filepath": str(audio_path),
                "file_size_mb": audio_path.stat().st_size / (1024 * 1024)
            }
    
    def _transcribe_with_whisper(self, audio_path: Path) -> Dict:
        """
        Transcribe audio using Whisper.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Whisper transcription result with segments
        """
        try:
            result = self.whisper_model.transcribe(
                str(audio_path),
                task="transcribe",
                language=None,  # Auto-detect
                word_timestamps=True,
                verbose=False
            )
            return result
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            raise
    
    def _extract_segments(self, transcription_result: Dict) -> List[TranscriptSegment]:
        """
        Extract transcript segments from Whisper result.
        
        Args:
            transcription_result: Whisper transcription output
            
        Returns:
            List of TranscriptSegment objects
        """
        segments = []
        
        for segment in transcription_result.get("segments", []):
            transcript_segment = TranscriptSegment(
                text=segment["text"].strip(),
                start_time=segment["start"],
                end_time=segment["end"],
                confidence=segment.get("no_speech_prob", None)
            )
            segments.append(transcript_segment)
        
        return segments
    
    def _apply_diarization(
        self,
        audio_path: Path,
        segments: List[TranscriptSegment]
    ) -> List[TranscriptSegment]:
        """
        Apply speaker diarization to transcript segments.
        
        Args:
            audio_path: Path to audio file
            segments: List of transcript segments
            
        Returns:
            Segments with speaker labels
        """
        try:
            # Run diarization
            diarization = self.diarization_pipeline(str(audio_path))
            
            # Create speaker timeline
            speaker_timeline = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_timeline.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker
                })
            
            # Assign speakers to segments
            for segment in segments:
                segment.speaker = self._find_speaker(
                    segment.start_time,
                    segment.end_time,
                    speaker_timeline
                )
            
            logger.info(f"Diarization complete. Speakers found: {self._count_speakers(segments)}")
            return segments
            
        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            # Return segments without speaker labels
            return segments
    
    def _find_speaker(
        self,
        start_time: float,
        end_time: float,
        speaker_timeline: List[Dict]
    ) -> str:
        """
        Find the speaker for a given time range.
        
        Args:
            start_time: Segment start time
            end_time: Segment end time
            speaker_timeline: List of speaker turns
            
        Returns:
            Speaker label
        """
        # Find overlapping speaker turns
        overlaps = []
        segment_duration = end_time - start_time
        
        for turn in speaker_timeline:
            overlap_start = max(start_time, turn["start"])
            overlap_end = min(end_time, turn["end"])
            overlap_duration = max(0, overlap_end - overlap_start)
            
            if overlap_duration > 0:
                overlaps.append({
                    "speaker": turn["speaker"],
                    "overlap": overlap_duration
                })
        
        # Return speaker with most overlap
        if overlaps:
            best_match = max(overlaps, key=lambda x: x["overlap"])
            return best_match["speaker"]
        
        return "UNKNOWN"
    
    def _count_speakers(self, segments: List[TranscriptSegment]) -> int:
        """Count unique speakers in segments."""
        speakers = set(seg.speaker for seg in segments if seg.speaker)
        return len(speakers)
    
    def _save_results(
        self,
        audio_path: Path,
        segments: List[TranscriptSegment],
        metadata: Dict,
        output_dir: str
    ):
        """
        Save transcription results to files.
        
        Args:
            audio_path: Original audio file path
            segments: Transcript segments
            metadata: Audio metadata
            output_dir: Output directory
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            base_name = audio_path.stem
            
            # Save JSON with full details
            json_path = output_path / f"{base_name}_transcript.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "metadata": metadata,
                    "segments": [seg.to_dict() for seg in segments],
                    "full_text": " ".join(seg.text for seg in segments)
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved JSON transcript: {json_path}")
            
            # Save human-readable text with timestamps
            txt_path = output_path / f"{base_name}_transcript.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"Transcription: {audio_path.name}\n")
                f.write(f"Duration: {metadata.get('duration_seconds', 'N/A')}s\n")
                f.write("=" * 80 + "\n\n")
                
                for segment in segments:
                    speaker_label = f"[{segment.speaker}] " if segment.speaker else ""
                    f.write(f"[{segment.get_timestamp_str()}] {speaker_label}{segment.text}\n")
            
            logger.info(f"Saved text transcript: {txt_path}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")


def ingest_audio_file(audio_path: str, output_dir: Optional[str] = None) -> Tuple[List[TranscriptSegment], Dict]:
    """
    Convenience function to ingest a single audio file.
    
    Args:
        audio_path: Path to audio file
        output_dir: Optional output directory
        
    Returns:
        Tuple of (segments, metadata)
    """
    pipeline = AudioIngestionPipeline()
    return pipeline.ingest_audio(audio_path, output_dir)