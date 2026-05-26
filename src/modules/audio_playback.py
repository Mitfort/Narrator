import os
import threading
import pygame


class AudioPlayback:
    def __init__(
        self,
        mixer_settings: dict = None,
        sound_mapping: dict = None,
        max_duration: float = 5.0,
    ):
        self.mixer_settings = mixer_settings or {
            "frequency": 44100,
            "size": -16,
            "channels": 2,
            "buffer": 512,
        }
        self.sound_mapping = sound_mapping or {}
        self.max_duration = max_duration

        # Timer per channel — only for non-looping (event) sounds
        self._stop_timers: dict[int, threading.Timer] = {}

        pygame.mixer.init(**self.mixer_settings)
        self.channels: dict[int, pygame.mixer.Channel] = {}
        self._loaded_sounds: dict[str, pygame.mixer.Sound] = {}

        if self.sound_mapping:
            self._init_channels()
            self._preload_assets()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

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
                print(f"[AudioPlayback] Warning: missing file: {path}")

    # ------------------------------------------------------------------
    # Stop-timer helpers  (event sounds only)
    # ------------------------------------------------------------------

    def _cancel_stop_timer(self, channel_id: int):
        timer = self._stop_timers.pop(channel_id, None)
        if timer and timer.is_alive():
            timer.cancel()

    def _schedule_stop(self, channel_id: int, duration: float):
        """Schedule a fadeout for non-looping sounds after *duration* seconds."""
        self._cancel_stop_timer(channel_id)
        timer = threading.Timer(duration, self._stop_channel, args=[channel_id])
        timer.daemon = True
        self._stop_timers[channel_id] = timer
        timer.start()

    def _stop_channel(self, channel_id: int):
        channel = self.channels.get(channel_id)
        if channel and channel.get_busy():
            channel.fadeout(500)
        self._stop_timers.pop(channel_id, None)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def play_sound(self, file_path: str, channel_id: int, is_loop: bool):
        sound = self._loaded_sounds.get(file_path)
        channel = self.channels.get(channel_id)

        if not sound or not channel:
            print(
                f"[AudioPlayback] Cannot play '{file_path}' on channel {channel_id} "
                f"— sound or channel not found."
            )
            return

        # If the same sound is already looping on this channel, leave it running.
        if is_loop and channel.get_busy() and channel.get_sound() is sound:
            print(
                f"[AudioPlayback] '{file_path}' already looping on channel "
                f"{channel_id} — keeping."
            )
            return

        # Different sound on the same channel → fade out the old one first.
        if channel.get_busy():
            self._cancel_stop_timer(channel_id)
            channel.fadeout(500)

        print(
            f"[AudioPlayback] Playing '{file_path}' on channel {channel_id} "
            f"loop={is_loop}"
        )

        if is_loop:
            # Looping background: play indefinitely, no stop timer.
            channel.play(sound, loops=-1, fade_ms=500)
            self._schedule_stop(channel_id, self.max_duration * 2)
        else:
            # One-shot event: play once, schedule automatic cleanup.
            channel.play(sound, fade_ms=0)
            self._schedule_stop(channel_id, self.max_duration)

    def stop_channel(self, channel_id: int, fade_ms: int = 1000):
        """Explicitly stop a channel (e.g. when the scene changes)."""
        self._cancel_stop_timer(channel_id)
        channel = self.channels.get(channel_id)
        if channel and channel.get_busy():
            channel.fadeout(fade_ms)

    def stop_all(self, fade_ms: int = 1000):
        for timer in list(self._stop_timers.values()):
            if timer.is_alive():
                timer.cancel()
        self._stop_timers.clear()
        pygame.mixer.fadeout(fade_ms)