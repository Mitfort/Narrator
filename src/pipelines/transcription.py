import time
import math
import threading
import numpy as np
import json

from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future
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
    return np.sqrt(np.mean(audio ** 2))


class TranscriptionPipeline:
    def __init__(self, on_transcription: Callable[[str], None] | None = None):
        self.on_transcription = on_transcription

        config = get_config()

        self.playback_pipeline = PlaybackPipeline(config=config)
        self.model = WhisperModel(
            config["Whisper"]["Model"],
            device=config["Whisper"]["Device"],
            num_workers=config["Whisper"]["Num_Workers"],
            use_auth_token=config["Whisper"].get("Use_Auth_Token", False)
        )
        print("[TranscriptionPipeline] Initialized the model\n")

        fast_chunk_duration = config.get(
            "Fast_Chunk_Duration", config["Audio_Capture"]["Chunk_Duration"]
        )
        self.audio_capture = AudioCapture(
            sample_rate=config["Audio_Capture"]["Sample_Rate"],
            channels=config["Audio_Capture"]["Channels"],
            chunk_duration=fast_chunk_duration,
        )

        self.buffer: list[np.ndarray] = []
        self.buffer_duration: float = 0.0
        self.silence_sec: float = 0.0
        self.speaking: bool = False

        self._silence_threshold: float = config["VAD"]["Silence_Threshold"]
        self._silence_threshold_lock = threading.Lock()

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
        self.noise_raise_threshold_factor: float = config.get(
            "Noise_Raise_Threshold_Factor", 1.5
        )

        max_chunks = max(
            1,
            int(self.noise_window_sec / max(0.001, self.audio_capture.chunk_duration)),
        )
        self.noise_buffer: deque[float] = deque(maxlen=max_chunks)

        self.transcription_count: int = 0
        self.last_transcription_time: float = 0.0
        self.total_transcription_time: float = 0.0

        # Single-worker executor: one transcription at a time, non-blocking for audio loop
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="transcription"
        )
        self._pending: Future | None = None

    # ------------------------------------------------------------------
    # Properties with lock for silence_threshold (shared with bg thread)
    # ------------------------------------------------------------------

    @property
    def silence_threshold(self) -> float:
        with self._silence_threshold_lock:
            return self._silence_threshold

    @silence_threshold.setter
    def silence_threshold(self, value: float):
        with self._silence_threshold_lock:
            self._silence_threshold = value

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

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
            self._executor.shutdown(wait=True)
            self.playback_pipeline.close()

    # ------------------------------------------------------------------
    # Audio chunking / VAD  (main thread only)
    # ------------------------------------------------------------------

    def process_chunk(self, chunk: np.ndarray):
        chunk_rms = rms(chunk)
        silent = chunk_rms < self.silence_threshold

        # Collect ambient noise only while not speaking
        if not self.speaking and silent:
            self.noise_buffer.append(float(chunk_rms))

        if not silent:
            self.buffer.append(chunk)
            self.buffer_duration += self.audio_capture.chunk_duration
            self.silence_sec = 0.0
            self.speaking = True

            # Fast path: partial transcription every max_buffer_duration seconds
            if self.fast_enabled and self.buffer_duration >= self.max_buffer_duration:
                self._submit_buffer(keep_speaking=True)

        elif self.speaking:
            self.buffer.append(chunk)
            self.silence_sec += self.audio_capture.chunk_duration

            if self.silence_sec >= self.silence_duration:
                self._submit_buffer(keep_speaking=False)

    def _submit_buffer(self, keep_speaking: bool = False):
        """
        Snapshot the buffer and reset it immediately so that the audio-capture
        loop is never blocked by transcription.  The actual work runs in a
        background thread.
        """
        if not self.buffer:
            return

        # --- snapshot & reset BEFORE any async work ---
        audio_snapshot = np.concatenate(self.buffer, axis=0).flatten()
        noise_snapshot = list(self.noise_buffer)

        self.buffer = []
        self.buffer_duration = 0.0
        self.silence_sec = 0.0
        self.speaking = keep_speaking

        duration = len(audio_snapshot) / self.audio_capture.sample_rate
        if duration < self.min_audio_duration:
            return

        # Fast-path: drop if previous transcription is still running
        if self._pending is not None and not self._pending.done():
            if keep_speaking:
                print(
                    "[TranscriptionPipeline] Fast-path flush skipped — "
                    "previous transcription still running."
                )
                return
            # End-of-speech: wait for the previous job before submitting the
            # final segment so results arrive in order.
            self._pending.result()

        self._pending = self._executor.submit(
            self._process_audio, audio_snapshot, noise_snapshot, duration
        )

    # ------------------------------------------------------------------
    # Background thread work
    # ------------------------------------------------------------------

    def _process_audio(
        self, audio: np.ndarray, noise_snapshot: list[float], duration: float
    ):
        """Transcribe → SNR check → hotword detection → playback → callback."""
        print(
            f"[TranscriptionPipeline] Processing audio chunk "
            f"of duration {duration:.2f}s..."
        )

        start_time = time.perf_counter()
        text = self._transcribe(audio)
        elapsed = time.perf_counter() - start_time

        self.transcription_count += 1
        self.last_transcription_time = elapsed
        self.total_transcription_time += elapsed
        avg = self.total_transcription_time / self.transcription_count
        ratio = elapsed / duration if duration > 0 else float("inf")

        print(
            f"[TranscriptionPipeline] Transcription #{self.transcription_count}: "
            f"audio {duration:.2f}s, process {elapsed:.2f}s, "
            f"ratio {ratio:.2f}x, avg {avg:.2f}s"
        )

        if not text:
            return

        # --- SNR check ---
        audio_rms = rms(audio)
        ambient = float(np.median(noise_snapshot)) if noise_snapshot else 0.0
        eps = 1e-9
        snr_db = (
            20.0 * math.log10((audio_rms + eps) / (ambient + eps))
            if ambient > 0.0
            else float("inf")
        )
        snr_label = f"{snr_db:.2f}" if snr_db != float("inf") else "inf"
        print(
            f"[TranscriptionPipeline] ambient={ambient:.6f}, "
            f"audio_rms={audio_rms:.6f}, snr_db={snr_label}"
        )

        low_snr = snr_db != float("inf") and snr_db < self.noise_snr_threshold_db
        if low_snr:
            print(
                f"[TranscriptionPipeline] Low SNR ({snr_label} dB < "
                f"{self.noise_snr_threshold_db} dB)."
            )
            if self.noise_skip_on_low_snr:
                print(
                    "[TranscriptionPipeline] Skipping — high background noise."
                )
                return
            # Temporarily raise VAD threshold for future chunks while processing
            old_thresh = self.silence_threshold
            self.silence_threshold = old_thresh * self.noise_raise_threshold_factor
            print(
                f"[TranscriptionPipeline] Temporarily raising silence threshold "
                f"to {self.silence_threshold:.6f}."
            )
            self._dispatch(text)
            self.silence_threshold = old_thresh
        else:
            self._dispatch(text)

    def _dispatch(self, text: str):
        """
        Check for hotwords and fire on_transcription exactly once with the
        most relevant version of the text.

        - Hotword found  → process_text() triggers playback and may correct
                           the text; callback receives the corrected version.
        - No hotword     → callback receives the original transcription.
        """
        if self.hotword_detector.has_hotword(text):
            corrected_text = self.playback_pipeline.process_text(text)
            if self.on_transcription:
                self.on_transcription(corrected_text)
        else:
            print("[TranscriptionPipeline] No hotword detected in this segment.")
            if self.on_transcription:
                self.on_transcription(text)

    def _transcribe(self, audio: np.ndarray) -> str:
        beam_size = self.fast_beam_size if self.fast_enabled else 5
        segments, _ = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=beam_size,
            vad_filter=True,
        )
        return " ".join(seg.text for seg in segments).strip()