import string
from src.modules.audio_playback import AudioPlayback

class PlaybackPipeline:
    def __init__(self, config: dict):
        self.config = config
        mixer_settings = self.config.get("Playback_Mixer")
        sound_mapping = self._parse_mappings()
        self.playback = AudioPlayback(mixer_settings, sound_mapping)

    def _parse_mappings(self) -> dict:
        channels_map = self.config.get("Playback_Channels", {})
        files_map = self.config.get("Playback_Sound_Files", {})
        raw_mapping = self.config.get("Playback_Sound_Mapping", {})
        
        parsed_mapping = {}
        for hotword, config in raw_mapping.items():
            parsed_mapping[hotword] = {
                "file": files_map.get(config["file_key"]),
                "is_loop": config["is_loop"],
                "channel": channels_map.get(config["channel_key"])
            }
        return parsed_mapping

    def process_text(self, text: str):
        translator = str.maketrans('', '', string.punctuation)
        words = text.translate(translator).lower().split()
        
        for word in words:
            if word in self.playback.sound_mapping:
                config = self.playback.sound_mapping[word]
                print(f"[PlaybackPipeline] Triggered hotword: '{word}'")
                self.playback.play_sound(
                    file_path=config["file"],
                    channel_id=config["channel"],
                    is_loop=config["is_loop"]
                )

    def close(self):
        self.playback.stop_all()