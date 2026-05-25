import time
import numpy as np
import json

from typing import Callable
from pathlib import Path
from faster_whisper import WhisperModel
from src.modules.audio_capture import AudioCapture

from src.pipelines.playback import PlaybackPipeline

ROOT_DIR = Path(__file__).parent.parent

def get_config():
    config_dir = ROOT_DIR / "utils"
    with open(config_dir / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    def _load_json_from_path(key: str) -> dict:
        path_value = config.get(f"{key}_Path")
        if not isinstance(path_value, str):
            return config.get(key, {})

        path = Path(path_value)
        if not path.is_absolute():
            path = config_dir / path

        try:
            with open(path, "r", encoding="utf-8") as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"[get_config] File not found for {key}_Path: {path}")
        except Exception as exc:
            print(f"[get_config] Error loading {key}_Path {path}: {exc}")

        return config.get(key, {})

    config["Playback_Channels"] = _load_json_from_path("Playback_Channels")
    config["Playback_Sound_Files"] = _load_json_from_path("Playback_Sound_Files")
    config["Playback_Sound_Mapping"] = _load_json_from_path("Playback_Sound_Mapping")

    return config

def rms(audio: np.ndarray) -> float:
    return np.sqrt(np.mean(audio**2))

class TranscriptionPipeline:
    def __init__(self, on_transcription: Callable[[str], None] | None = None):
        self.on_transcription = on_transcription
        self.audio_capture = AudioCapture()

        config = get_config()

        self.playback_pipeline = PlaybackPipeline(config=config)
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

        self.transcription_count: int = 0
        self.last_transcription_time: float = 0.0
        self.total_transcription_time: float = 0.0

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
            self.playback_pipeline.close()

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

            start_time = time.perf_counter()
            text = self.transcribe(audio)
            end_time = time.perf_counter()

            transcription_time = end_time - start_time
            self.transcription_count += 1
            self.last_transcription_time = transcription_time
            self.total_transcription_time += transcription_time
            average_time = self.total_transcription_time / self.transcription_count
            realtime_ratio = transcription_time / duration if duration > 0 else float('inf')

            print(
                f"[TranscriptionPipeline] Transcription #{self.transcription_count}: "
                f"audio {duration:.2f}s, process {transcription_time:.2f}s, "
                f"ratio {realtime_ratio:.2f}x, avg {average_time:.2f}s"
            )

            if text:
                corrected_text = self.playback_pipeline.process_text(text)
                
                if self.on_transcription:
                    self.on_transcription(corrected_text)
        
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