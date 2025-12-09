"""
Audio Ingestion with AssemblyAI
Better speaker diarization alternative to pyannote.audio
"""

import os
from typing import List, Optional
from pathlib import Path
from loguru import logger

try:
    import assemblyai as aai
    ASSEMBLYAI_AVAILABLE = True
except ImportError:
    ASSEMBLYAI_AVAILABLE = False
    logger.warning("AssemblyAI not installed. Install with: pip install assemblyai")

from .audio_ingestion import TranscriptSegment


class AssemblyAIAudioPipeline:
    """
    Audio ingestion using AssemblyAI for transcription and speaker diarization.
    More reliable than pyannote.audio with better accuracy.
    """
    
    def __init__(self, enable_diarization: bool = True):
        """
        Initialize AssemblyAI pipeline.
        
        Args:
            enable_diarization: Enable speaker diarization
        """
        if not ASSEMBLYAI_AVAILABLE:
            raise ImportError("AssemblyAI not installed. Install with: pip install assemblyai")
        
        api_key = os.getenv("ASSEMBLYAI_API_KEY")
        if not api_key or api_key == "your_assemblyai_api_key_here":
            raise ValueError(
                "ASSEMBLYAI_API_KEY not set in .env\n"
                "Get your API key from: https://www.assemblyai.com/\n"
                "Free tier: 5 hours/month"
            )
        
        aai.settings.api_key = api_key
        self.enable_diarization = enable_diarization
        
        logger.info(f"AssemblyAI audio pipeline initialized (diarization={'enabled' if enable_diarization else 'disabled'})")
    
    def ingest_audio(
        self,
        audio_path: str,
        output_dir: Optional[str] = None,
        num_speakers: Optional[int] = None
    ) -> List[TranscriptSegment]:
        """
        Transcribe and diarize audio using AssemblyAI.
        
        Args:
            audio_path: Path to audio file
            output_dir: Directory to save output (optional)
            num_speakers: Expected number of speakers (optional, helps accuracy)
            
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
                speaker_labels=self.enable_diarization,
                speakers_expected=num_speakers  # Optional: helps accuracy if known
            )
            
            # Create transcriber
            transcriber = aai.Transcriber()
            
            # Transcribe
            logger.info("Uploading and transcribing audio...")
            transcript = transcriber.transcribe(str(audio_path), config=config)
            
            # Check for errors
            if transcript.status == aai.TranscriptStatus.error:
                raise Exception(f"AssemblyAI transcription failed: {transcript.error}")
            
            # Convert to TranscriptSegment objects
            segments = []
            
            if self.enable_diarization and transcript.utterances:
                # Use utterances (includes speaker labels)
                for utterance in transcript.utterances:
                    segment = TranscriptSegment(
                        text=utterance.text,
                        start_time=utterance.start / 1000.0,  # Convert ms to seconds
                        end_time=utterance.end / 1000.0,
                        speaker=f"SPEAKER_{utterance.speaker}",
                        confidence=utterance.confidence if hasattr(utterance, 'confidence') else None
                    )
                    segments.append(segment)
                
                # Count unique speakers
                speakers = set(s.speaker for s in segments if s.speaker)
                logger.info(f"✅ Transcribed {len(segments)} utterances with {len(speakers)} speakers")
                logger.info(f"   Speakers: {sorted(speakers)}")
            else:
                # Use words/sentences (no speaker labels)
                if transcript.words:
                    # Group words into segments by sentence
                    current_segment = []
                    current_start = None
                    
                    for i, word in enumerate(transcript.words):
                        if current_start is None:
                            current_start = word.start
                        
                        current_segment.append(word.text)
                        
                        # End segment on punctuation or every ~30 words
                        is_end = (
                            word.text.endswith(('.', '!', '?')) or 
                            len(current_segment) >= 30 or
                            i == len(transcript.words) - 1
                        )
                        
                        if is_end:
                            segment = TranscriptSegment(
                                text=' '.join(current_segment),
                                start_time=current_start / 1000.0,
                                end_time=word.end / 1000.0,
                                speaker="Unknown",
                                confidence=word.confidence if hasattr(word, 'confidence') else None
                            )
                            segments.append(segment)
                            current_segment = []
                            current_start = None
                
                logger.info(f"✅ Transcribed {len(segments)} segments (no speaker labels)")
            
            # Save output if directory specified
            if output_dir:
                self._save_segments(segments, audio_path.name, output_dir)
            
            logger.info(f"✅ Audio ingestion complete: {len(segments)} segments")
            
            return segments
            
        except Exception as e:
            logger.error(f"AssemblyAI transcription failed: {e}", exc_info=True)
            raise
    
    def _save_segments(
        self,
        segments: List[TranscriptSegment],
        filename: str,
        output_dir: str
    ):
        """Save transcription segments to JSON file."""
        import json
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{Path(filename).stem}_transcript_assemblyai.json"
        
        # Prepare data
        speakers = list(set(s.speaker for s in segments if s.speaker))
        
        output_data = {
            "source_file": filename,
            "num_segments": len(segments),
            "total_duration": segments[-1].end_time if segments else 0,
            "speakers": speakers,
            "num_speakers": len(speakers),
            "transcription_service": "AssemblyAI",
            "segments": [s.to_dict() for s in segments]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Saved transcript to: {output_file}")