import os
import pygame

class AudioPlayback:
    def __init__(self, mixer_settings: dict = None, sound_mapping: dict = None):
        self.mixer_settings = mixer_settings or {
            "frequency": 44100, "size": -16, "channels": 2, "buffer": 512
        }
        self.sound_mapping = sound_mapping or {}
        
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

    def play_sound(self, file_path: str, channel_id: int, is_loop: bool):
        sound = self._loaded_sounds.get(file_path)
        channel = self.channels.get(channel_id)
        
        if not sound or not channel:
            return

        if is_loop:
            if not channel.get_busy() or channel.get_sound() != sound:
                channel.play(sound, loops=-1, fade_ms=2000)
        else:
            channel.play(sound)

    def stop_all(self):
        pygame.mixer.fadeout(1000)