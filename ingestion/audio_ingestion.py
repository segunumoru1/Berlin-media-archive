"""
Audio Ingestion Pipeline using Whisper and Speaker Diarization
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import whisper
from loguru import logger

# Use our custom audio utils instead of pydub
from utils.audio_utils import get_audio_duration

@dataclass
class TranscriptSegment:
    """Transcript segment with metadata"""
    text: str
    start_time: float
    end_time: float
    speaker: Optional[str] = None
    confidence: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


class AudioIngestionPipeline:
    """Audio ingestion with Whisper transcription and speaker diarization"""
    
    def __init__(
        self,
        whisper_model: str = "base",
        enable_diarization: bool = True,
        chunk_length_seconds: int = 30
    ):
        self.whisper_model_name = whisper_model
        self.enable_diarization = enable_diarization
        self.chunk_length_ms = chunk_length_seconds * 1000
        
        logger.info(f"Loading Whisper model: {whisper_model}")
        self.whisper_model = whisper.load_model(whisper_model)
        
        self.diarization_pipeline = None
        if enable_diarization:
            try:
                from pyannote.audio import Pipeline
                hf_token = os.getenv("HUGGINGFACE_TOKEN")
                if hf_token:
                    self.diarization_pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1",
                        token=hf_token
                    )
                    logger.info("Speaker diarization enabled")
                else:
                    logger.warning("HUGGINGFACE_TOKEN not set, diarization disabled")
            except Exception as e:
                logger.warning(f"Could not load diarization pipeline: {e}")
    
    def transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe audio using Whisper"""
        logger.info(f"Transcribing: {audio_path}")
        
        result = self.whisper_model.transcribe(
            audio_path,
            language="en",
            task="transcribe",
            verbose=False
        )
        
        return result
    
    def perform_diarization(self, audio_path: str) -> Optional[Dict]:
        """Perform speaker diarization"""
        if not self.diarization_pipeline:
            return None
        
        try:
            logger.info("Performing speaker diarization...")
            diarization = self.diarization_pipeline(audio_path)
            
            speaker_segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_segments.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker
                })
            
            return {"segments": speaker_segments}
        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            return None
    
    def merge_transcription_with_speakers(
        self,
        transcription: Dict[str, Any],
        diarization: Optional[Dict]
    ) -> List[TranscriptSegment]:
        """Merge Whisper transcription with speaker labels"""
        segments = []
        
        for segment in transcription.get("segments", []):
            speaker = None
            
            if diarization:
                mid_time = (segment["start"] + segment["end"]) / 2
                for spk_seg in diarization.get("segments", []):
                    if spk_seg["start"] <= mid_time <= spk_seg["end"]:
                        speaker = spk_seg["speaker"]
                        break
            
            segments.append(TranscriptSegment(
                text=segment["text"].strip(),
                start_time=segment["start"],
                end_time=segment["end"],
                speaker=speaker,
                confidence=segment.get("confidence")
            ))
        
        return segments
    
    def ingest_audio(
        self,
        audio_path: str,
        output_dir: str
    ) -> Dict[str, Any]:
        """Main ingestion pipeline"""
        try:
            audio_file = Path(audio_path)
            if not audio_file.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_file}")
            
            logger.info(f"Processing audio: {audio_file}")
            
            # Get audio duration
            duration = get_audio_duration(str(audio_file))
            
            # Transcribe
            transcription = self.transcribe_audio(str(audio_file))
            
            # Diarization (optional)
            diarization = None
            if self.enable_diarization and self.diarization_pipeline:
                diarization = self.perform_diarization(str(audio_file))
            
            # Merge results
            segments = self.merge_transcription_with_speakers(
                transcription,
                diarization
            )
            
            # Extract unique speakers
            speakers = list(set(
                seg.speaker for seg in segments if seg.speaker
            ))
            
            # Save results
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            output_file = output_path / f"{audio_file.stem}_transcript.json"
            
            result = {
                "source_file": str(audio_path),
                "duration_seconds": duration,
                "num_segments": len(segments),
                "speakers": speakers,
                "segments": [seg.to_dict() for seg in segments],
                "full_text": " ".join(seg.text for seg in segments)
            }
            
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved transcript to: {output_file}")
            
            return result
            
        except Exception as e:
            logger.error(f"Audio ingestion failed: {e}", exc_info=True)
            raise

    