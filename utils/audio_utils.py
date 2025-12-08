"""
Audio Utilities - Python 3.13 Compatible
Replacement for pydub using soundfile and librosa
"""

import soundfile as sf
import librosa
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import io


class AudioSegment:
    """Python 3.13 compatible audio segment handler"""
    
    def __init__(self, audio_data: np.ndarray, sample_rate: int):
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.duration = len(audio_data) / sample_rate
    
    @classmethod
    def from_file(cls, file_path: str) -> 'AudioSegment':
        """Load audio from file"""
        audio_data, sample_rate = librosa.load(file_path, sr=None, mono=True)
        return cls(audio_data, int(sample_rate))
    
    @classmethod
    def from_wav(cls, file_path: str) -> 'AudioSegment':
        """Load WAV file"""
        audio_data, sample_rate = sf.read(file_path)
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)  # Convert to mono
        return cls(audio_data, sample_rate)
    
    @classmethod
    def from_mp3(cls, file_path: str) -> 'AudioSegment':
        """Load MP3 file"""
        audio_data, sample_rate = librosa.load(file_path, sr=None, mono=True)
        return cls(audio_data, int(sample_rate))
    
    def export(self, file_path: str, format: str = "wav") -> str:
        """Export audio to file"""
        sf.write(file_path, self.audio_data, self.sample_rate, format=format.upper())
        return file_path
    
    def get_array_of_samples(self) -> np.ndarray:
        """Get raw audio samples"""
        return self.audio_data
    
    def set_frame_rate(self, new_rate: int) -> 'AudioSegment':
        """Resample audio"""
        resampled = librosa.resample(
            self.audio_data,
            orig_sr=self.sample_rate,
            target_sr=new_rate
        )
        return AudioSegment(resampled, new_rate)
    
    def __len__(self) -> int:
        """Duration in milliseconds"""
        return int(self.duration * 1000)
    
    def __getitem__(self, milliseconds: slice) -> 'AudioSegment':
        """Slice audio by milliseconds"""
        start_sample = int((milliseconds.start or 0) * self.sample_rate / 1000)
        stop_sample = int((milliseconds.stop or len(self)) * self.sample_rate / 1000)
        
        sliced_data = self.audio_data[start_sample:stop_sample]
        return AudioSegment(sliced_data, self.sample_rate)
    
    @property
    def frame_rate(self) -> int:
        """Get sample rate"""
        return self.sample_rate
    
    @property
    def channels(self) -> int:
        """Number of channels (always 1 for mono)"""
        return 1


def convert_audio_format(input_path: str, output_path: str, output_format: str = "wav") -> str:
    """Convert audio between formats"""
    audio = AudioSegment.from_file(input_path)
    return audio.export(output_path, format=output_format)


def get_audio_duration(file_path: str) -> float:
    """Get audio duration in seconds"""
    audio = AudioSegment.from_file(file_path)
    return audio.duration


def split_audio_chunks(file_path: str, chunk_length_ms: int = 30000) -> list:
    """Split audio into chunks"""
    audio = AudioSegment.from_file(file_path)
    chunks = []
    
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        chunks.append(chunk)
    
    return chunks