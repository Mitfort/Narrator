import os
import threading
import pygame

class AudioPlayback:
    def __init__(self, mixer_settings: dict = None, sound_mapping: dict = None, max_duration: float = 5.0):
        self.mixer_settings = mixer_settings or {
            "frequency": 44100, "size": -16, "channels": 2, "buffer": 512
        }
        self.sound_mapping = sound_mapping or {}
        self.max_duration = max_duration
        self._stop_timers: dict[int, threading.Timer] = {}

        pygame.mixer.init(**self.mixer_settings)
        self.channels = {}
        self._loaded_sounds = {}
        
        if self.sound_mapping:
            self._init_channels()
            self._preload_assets()

    def _init_channels(self):
        for config in self.sound_mapping.values():
            ch_id = config["channel"]
            if ch_id not in self.channels:
                self.channels[ch_id] = pygame.mixer.Channel(ch_id)

    def _preload_assets(self):
        unique_paths = {config["file"] for config in self.sound_mapping.values()}
        for path in unique_paths:
            if os.path.exists(path):
                self._loaded_sounds[path] = pygame.mixer.Sound(path)
            else:
                print(f"[AudioPlayback Warning] Brak pliku: {path}")

    def _cancel_stop_timer(self, channel_id: int):
        timer = self._stop_timers.pop(channel_id, None)
        if timer and timer.is_alive():
            timer.cancel()

    def _schedule_stop(self, channel_id: int, duration: float):
        self._cancel_stop_timer(channel_id)
        timer = threading.Timer(duration, lambda: self._stop_channel(channel_id))
        self._stop_timers[channel_id] = timer
        timer.daemon = True
        timer.start()

    def _stop_channel(self, channel_id: int):
        channel = self.channels.get(channel_id)
        if channel and channel.get_busy():
            channel.fadeout(500)
        self._stop_timers.pop(channel_id, None)

    def play_sound(self, file_path: str, channel_id: int, is_loop: bool):
        sound = self._loaded_sounds.get(file_path)
        channel = self.channels.get(channel_id)
        
        if not sound or not channel:
            return

        current_sound = channel.get_sound()
        if channel.get_busy() and current_sound is not None and current_sound != sound:
            channel.fadeout(500)

        self._cancel_stop_timer(channel_id)
        print(f"[AudioPlayback] Playing file '{file_path}' on channel {channel_id} loop={is_loop}")

        if is_loop:
            channel.play(sound, loops=-1, fade_ms=2000)
        else:
            channel.play(sound)

        self._schedule_stop(channel_id, self.max_duration)

    def stop_all(self):
        for timer in list(self._stop_timers.values()):
            if timer.is_alive():
                timer.cancel()
        self._stop_timers.clear()
        pygame.mixer.fadeout(1000)