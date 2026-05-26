import time
import math
import numpy as np
import json

from collections import deque
from typing import Callable
from pathlib import Path
from faster_whisper import WhisperModel
from src.modules.audio_capture import AudioCapture

from src.pipelines.hotword_detector import HotwordDetector
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

        fast_chunk_duration = config.get("Fast_Chunk_Duration", config["Audio_Capture"]["Chunk_Duration"])
        self.audio_capture = AudioCapture(
            sample_rate=config["Audio_Capture"]["Sample_Rate"],
            channels=config["Audio_Capture"]["Channels"],
            chunk_duration=fast_chunk_duration
        )

        self.buffer: list[np.ndarray] = []
        self.buffer_duration: float = 0.0
        self.silence_sec: float = 0.0
        self.speaking: bool = False
        self.silence_threshold: float = config["VAD"]["Silence_Threshold"]
        self.silence_duration: float = config["VAD"]["Silence_Duration"]
        self.min_audio_duration: float = config["VAD"]["Min_Audio_Duration"]
        self.language: str = config["Whisper"]["Language"]

        self.fast_enabled: bool = config.get("Fast_Path_Enabled", True)
        self.max_buffer_duration: float = config.get("Fast_Max_Buffer_Duration", 2.0)
        self.fast_beam_size: int = config.get("Fast_Beam_Size", 1)

        self.hotword_detector = HotwordDetector(config)

        self.noise_window_sec: float = config.get("Noise_Background_Window", 5.0)
        self.noise_snr_threshold_db: float = config.get("Noise_SNR_Threshold_DB", 6.0)
        self.noise_skip_on_low_snr: bool = config.get("Noise_Skip_On_Low_SNR", True)
        self.noise_raise_threshold_factor: float = config.get("Noise_Raise_Threshold_Factor", 1.5)

        max_chunks = max(1, int(self.noise_window_sec / max(0.001, self.audio_capture.chunk_duration)))
        self.noise_buffer: deque[float] = deque(maxlen=max_chunks)

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
        chunk_rms = rms(chunk)
        silent = chunk_rms < self.silence_threshold

        if not self.speaking and silent:
            try:
                self.noise_buffer.append(float(chunk_rms))
            except Exception:
                pass

        if not silent:
            self.buffer.append(chunk)
            self.buffer_duration += self.audio_capture.chunk_duration
            self.silence_sec = 0.0
            self.speaking = True

            if self.fast_enabled and self.buffer_duration >= self.max_buffer_duration:
                self.flush_buffer(keep_speaking=True)
        elif self.speaking:
            self.buffer.append(chunk)
            self.silence_sec += self.audio_capture.chunk_duration

            if self.silence_sec >= self.silence_duration:
                self.flush_buffer()
    
    def flush_buffer(self, keep_speaking: bool = False):
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

            # compute SNR vs ambient noise
            audio_rms = rms(audio)
            ambient = float(np.median(list(self.noise_buffer))) if self.noise_buffer else 0.0
            eps = 1e-9
            if ambient <= 0.0:
                snr_db = float('inf')
            else:
                snr_db = 20.0 * math.log10((audio_rms + eps) / (ambient + eps))

            print(f"[TranscriptionPipeline] ambient={ambient:.6f}, audio_rms={audio_rms:.6f}, snr_db={snr_db if snr_db!=float('inf') else 'inf'}")

            if snr_db != float('inf') and snr_db < self.noise_snr_threshold_db:
                print(f"[TranscriptionPipeline] Low SNR ({snr_db:.2f} dB) below threshold {self.noise_snr_threshold_db} dB.")
                if self.noise_skip_on_low_snr:
                    print("[TranscriptionPipeline] Skipping transcription due to high background noise.")
                else:
                    # raise current silence threshold temporarily and perform transcription
                    old_thresh = self.silence_threshold
                    self.silence_threshold = old_thresh * self.noise_raise_threshold_factor
                    print(f"[TranscriptionPipeline] Raising silence threshold to {self.silence_threshold:.6f} for this transcription.")
                    if text:
                        if self.hotword_detector.has_hotword(text):
                            corrected_text = self.playback_pipeline.process_text(text)
                            if self.on_transcription:
                                self.on_transcription(corrected_text)
                        else:
                            print("[TranscriptionPipeline] No hotword detected in this segment.")
                    # restore
                    self.silence_threshold = old_thresh
            else:
                if text:
                    self.on_transcription(text)
                    if self.hotword_detector.has_hotword(text):
                        corrected_text = self.playback_pipeline.process_text(text)
                        if self.on_transcription:
                            self.on_transcription(corrected_text)
                    else:
                        print("[TranscriptionPipeline] No hotword detected in this segment.")
        
        self.buffer = []
        self.buffer_duration = 0.0
        self.silence_sec = 0.0
        self.speaking = keep_speaking

    def transcribe(self, audio: np.ndarray) -> str:
        beam_size = self.fast_beam_size if self.fast_enabled else 5
        segments, info = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=beam_size,
            vad_filter=True,
        )

        segments = list(segments)
        text = " ".join(segment.text for segment in segments)
        return text.strip()