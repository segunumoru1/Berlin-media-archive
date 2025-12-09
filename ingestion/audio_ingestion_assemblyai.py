"""
Audio Ingestion with AssemblyAI
Provides better speaker diarization than pyannote.audio
"""
import os
from typing import List, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from loguru import logger

try:
    import assemblyai as aai
    ASSEMBLYAI_AVAILABLE = True
except ImportError:
    ASSEMBLYAI_AVAILABLE = False
    logger.warning("AssemblyAI not installed. Install with: pip install assemblyai")


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


class AssemblyAIAudioPipeline:
    """
    Audio ingestion using AssemblyAI for transcription and speaker diarization.
    More accurate than pyannote.audio and easier to use.
    """
    
    def __init__(self, enable_diarization: bool = True):
        """
        Initialize AssemblyAI audio pipeline.
        
        Args:
            enable_diarization: Enable speaker diarization
        """
        if not ASSEMBLYAI_AVAILABLE:
            raise ImportError("AssemblyAI not installed. Install with: pip install assemblyai")
        
        api_key = os.getenv("ASSEMBLYAI_API_KEY")
        if not api_key or api_key == "your_assemblyai_api_key_here":
            raise ValueError(
                "ASSEMBLYAI_API_KEY not set in .env file. "
                "Get your API key from https://www.assemblyai.com/"
            )
        
        aai.settings.api_key = api_key
        self.enable_diarization = enable_diarization
        
        logger.info(f"AssemblyAI pipeline initialized (diarization={'enabled' if enable_diarization else 'disabled'})")
    
    def ingest_audio(
        self,
        audio_path: str,
        output_dir: Optional[str] = None
    ) -> List[TranscriptSegment]:
        """
        Transcribe and optionally diarize audio using AssemblyAI.
        
        Args:
            audio_path: Path to audio file
            output_dir: Directory to save processed output
            
        Returns:
            List of TranscriptSegment objects with speaker labels
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        logger.info(f"Transcribing with AssemblyAI: {audio_path.name}")
        logger.info(f"File size: {audio_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        try:
            # Configure transcription
            config = aai.TranscriptionConfig(
                speaker_labels=self.enable_diarization,  # Enable speaker diarization
                speakers_expected=None  # Auto-detect number of speakers
            )
            
            # Transcribe
            logger.info("Uploading and transcribing audio (this may take a few minutes)...")
            transcriber = aai.Transcriber()
            transcript = transcriber.transcribe(str(audio_path), config=config)
            
            # Check for errors
            if transcript.status == aai.TranscriptStatus.error:
                raise Exception(f"Transcription failed: {transcript.error}")
            
            # Convert to TranscriptSegment objects
            segments = []
            
            if self.enable_diarization and transcript.utterances:
                # Use utterances (includes speaker labels)
                logger.info("Processing utterances with speaker labels...")
                for utterance in transcript.utterances:
                    segment = TranscriptSegment(
                        text=utterance.text,
                        start_time=utterance.start / 1000.0,  # Convert ms to seconds
                        end_time=utterance.end / 1000.0,
                        speaker=f"SPEAKER_{utterance.speaker}",  # Format: SPEAKER_A, SPEAKER_B, etc.
                        confidence=utterance.confidence if hasattr(utterance, 'confidence') else None
                    )
                    segments.append(segment)
                
                # Count unique speakers
                speakers = set(s.speaker for s in segments if s.speaker)
                logger.info(f"✅ Identified {len(speakers)} unique speakers: {sorted(speakers)}")
            
            else:
                # Use words or fallback to full transcript
                logger.info("Processing without speaker diarization...")
                if transcript.words:
                    # Group words into segments (roughly 10 words per segment)
                    words_per_segment = 10
                    for i in range(0, len(transcript.words), words_per_segment):
                        word_group = transcript.words[i:i + words_per_segment]
                        text = " ".join(w.text for w in word_group)
                        
                        segment = TranscriptSegment(
                            text=text,
                            start_time=word_group[0].start / 1000.0,
                            end_time=word_group[-1].end / 1000.0,
                            speaker=None,
                            confidence=sum(w.confidence for w in word_group) / len(word_group)
                        )
                        segments.append(segment)
                else:
                    # Fallback: single segment
                    segment = TranscriptSegment(
                        text=transcript.text,
                        start_time=0.0,
                        end_time=0.0,
                        speaker=None,
                        confidence=transcript.confidence if hasattr(transcript, 'confidence') else None
                    )
                    segments.append(segment)
            
            # Save output if directory specified
            if output_dir:
                self.save_segments(segments, audio_path.name, output_dir)
            
            logger.info(f"✅ Audio ingestion complete: {len(segments)} segments")
            
            return segments
            
        except Exception as e:
            logger.error(f"AssemblyAI transcription failed: {e}", exc_info=True)
            raise
    
    def save_segments(
        self,
        segments: List[TranscriptSegment],
        filename: str,
        output_dir: str
    ):
        """Save transcription segments to JSON file."""
        import json
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{Path(filename).stem}_transcript.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump([seg.to_dict() for seg in segments], f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Saved transcript to: {output_file}")