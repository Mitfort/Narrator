import numpy as np
import json

from typing import Callable
from pathlib import Path
from faster_whisper import WhisperModel
from src.modules.audio_capture import AudioCapture

ROOT_DIR = Path(__file__).parent.parent

def get_config():
    with open(ROOT_DIR / "utils" / "config.json", "r") as f:
        return json.load(f)
    
# if __name__ == "__main__":
#     config = get_config()
#     # print(config)
#     # print("123")

def rms(audio: np.ndarray) -> float:
    return np.sqrt(np.mean(audio**2))

class TranscriptionPipeline:
    def __init__(self, on_transcription: Callable[[str], None] | None = None):
        self.on_transcription = on_transcription
        self.audio_capture = AudioCapture()

        config = get_config()
        self.model = WhisperModel(
            config["Whisper"]['Model'],
            device=config["Whisper"]['Device'],
            num_workers=config["Whisper"]['Num_Workers']
        )

        print("[TranscriptionPipeline] Initialized the model\n")

        self.audio_capture = AudioCapture(
            sample_rate=config["Audio_Capture"]["Sample_Rate"],
            channels=config["Audio_Capture"]["Channels"],
            chunk_duration=config["Audio_Capture"]["Chunk_Duration"]
        )

        self.buffer: list[np.ndarray] = []
        self.silence_sec: float = 0.0
        self.speaking: bool = False
        self.silence_threshold: float = config["VAD"]["Silence_Threshold"]
        self.silence_duration: float = config["VAD"]["Silence_Duration"]
        self.min_audio_duration: float = config["VAD"]["Min_Audio_Duration"]
        self.language: str = config["Whisper"]["Language"]

    def run(self):
        self.audio_capture.start()
        print("[TranscriptionPipeline] Running transcription pipeline...")

        try:
            while True:
                chunk = self.audio_capture.get_chunk()
                if chunk is None:
                    continue
                self.process_chunk(chunk)
        except KeyboardInterrupt:
            print("\n[TranscriptionPipeline] Stopping transcription pipeline...")
        finally:
            self.audio_capture.stop()

    def process_chunk(self, chunk: np.ndarray):
        silent = rms(chunk) < self.silence_threshold

        if not silent:
            self.buffer.append(chunk)
            self.silence_sec = 0.0
            self.speaking = True
        
        elif self.speaking:
            self.buffer.append(chunk)
            self.silence_sec += self.audio_capture.chunk_duration

            if self.silence_sec >= self.silence_duration:
                self.flush_buffer()
    
    def flush_buffer(self):
        if not self.buffer:
            return
        
        audio = np.concatenate(self.buffer, axis=0).flatten()
        duration = len(audio) / self.audio_capture.sample_rate

        if duration >= self.min_audio_duration:
            print(f"[TranscriptionPipeline] Processing audio chunk of duration {duration:.2f} seconds...")
            
            text = self.transcribe(audio)

            if text:
                self.on_transcription(text)
        
        self.buffer = []
        self.silence_sec = 0.0
        self.speaking = False

    def transcribe(self, audio: np.ndarray) -> str:
        segments,info = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=5,
            vad_filter=True,
        )

        segments = list(segments)
        text = " ".join(segment.text for segment in segments)
        return text.strip()


