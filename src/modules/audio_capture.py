import queue
import sounddevice as sd
import numpy as np

class AudioCapture:
    def __init__(self, sample_rate: int=16000, channels: int=1, chunk_duration: float =0.5):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        self.audio_queue = queue.Queue()
        self.stream: sd.InputStream | None = None

    def _callback(self, indata: np.ndarray, frames: int, time, status):
        if status:
            print(f"Audio capture error: {status}")
        self.audio_queue.put(indata.copy())

    def start(self):
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            blocksize=self.chunk_size,
            callback=self._callback,
            dtype=np.float32,
        )

        self.stream.start()

        print("[AudioCapture] Audio capture started.")
    
    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            print("[AudioCapture] Audio capture stopped.")
    
    def get_chunk(self, timeout: float = 1.0) -> np.ndarray | None:
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None