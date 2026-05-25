import difflib
import string
import unicodedata
from typing import Optional
from src.modules.audio_playback import AudioPlayback

class PlaybackPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.validation_threshold = self.config.get("Validation_Threshold", 0.75)

        self.transition_phrases = [
            self._normalize(phrase)
            for phrase in self.config.get("Playback_Transition_Phrases", ["po", "potem", "następnie", "wtedy", "po chwili", "w końcu"])
        ]
        self.overlap_phrases = [
            self._normalize(phrase)
            for phrase in self.config.get("Playback_Overlap_Phrases", ["jednocześnie", "w tym samym czasie", "razem", "naraz"])
        ]

        self.nlp: Optional[object] = None
        try:
            import spacy
            model_name = self.config.get("Lemma_Model", "pl_core_news_sm")
            self.nlp = spacy.load(model_name)
            print(f"[PlaybackPipeline] Loaded spaCy model: {model_name}")
        except Exception as e:
            print(f"[PlaybackPipeline] spaCy not available or model not installed: {e}")
            self.nlp = None

        mixer_settings = self.config.get("Playback_Mixer")
        sound_mapping = self._parse_mappings()
        self.mapping_keys = list(sound_mapping.keys())
        max_duration = self.config.get("Playback_Default_Max_Duration", 5)
        self.playback = AudioPlayback(mixer_settings, sound_mapping, max_duration=max_duration)

    def _normalize(self, s: str) -> str:
        normalized = unicodedata.normalize('NFKD', s)
        ascii_only = ''.join(ch for ch in normalized if not unicodedata.combining(ch))
        translator = str.maketrans('', '', string.punctuation)
        return ascii_only.translate(translator).lower().strip()

    def _lemma_phrase(self, phrase: str) -> str:
        """Return lemma-joined phrase if spaCy is available, otherwise normalized phrase."""
        if not self.nlp:
            return self._normalize(phrase)
        doc = self.nlp(phrase)
        return " ".join(token.lemma_.lower() for token in doc if token.text.strip())

    def _parse_mappings(self) -> dict:
        channels_map = self.config.get("Playback_Channels", {})
        files_map = self.config.get("Playback_Sound_Files", {})
        raw_mapping = self.config.get("Playback_Sound_Mapping", {})

        parsed_mapping: dict[str, dict] = {}

        for hotword, cfg in raw_mapping.items():
            entry = {
                "file": files_map.get(cfg["file_key"]),
                "is_loop": cfg["is_loop"],
                "channel": channels_map.get(cfg["channel_key"]),
                "category": cfg.get("category", "background" if cfg["is_loop"] else "event"),
                "group": cfg.get("group")
            }

            key_norm = self._normalize(hotword)
            parsed_mapping[key_norm] = entry

            key_lemma = self._lemma_phrase(hotword)
            if key_lemma and key_lemma != key_norm:
                parsed_mapping[key_lemma] = entry

        return parsed_mapping

    def _find_hotword_matches(self, normalized_text: str, lemma_text: str | None) -> list[tuple[str, dict, int]]:
        matches: list[tuple[str, dict, int]] = []
        padded_norm = f" {normalized_text} "
        padded_lemma = f" {lemma_text} " if lemma_text else None

        for hotkey, cfg in self.playback.sound_mapping.items():
            search_key = f" {hotkey} "
            position = padded_norm.find(search_key)
            if position == -1 and padded_lemma:
                position = padded_lemma.find(search_key)

            if position == -1:
                position = self._find_fuzzy_phrase_position(normalized_text, hotkey)
                if position == -1 and padded_lemma:
                    position = self._find_fuzzy_phrase_position(lemma_text, hotkey)

            if position == -1:
                continue

            matches.append((hotkey, cfg, position))

        return sorted(matches, key=lambda item: item[2])

    def _find_fuzzy_phrase_position(self, text: str | None, hotkey: str) -> int:
        if not text:
            return -1

        tokens = text.split()
        hotkey_tokens = hotkey.split()
        if len(tokens) < len(hotkey_tokens):
            return -1

        best_ratio = 0.0
        best_position = -1
        for start in range(len(tokens) - len(hotkey_tokens) + 1):
            window = " ".join(tokens[start:start + len(hotkey_tokens)])
            ratio = difflib.SequenceMatcher(None, window, hotkey).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_position = start

        if best_ratio >= self.validation_threshold:
            # compute character position for the matching window
            normalized_window = " ".join(tokens[best_position:best_position + len(hotkey_tokens)])
            pos = text.find(normalized_window)
            return pos if pos != -1 else best_position

        return -1

    def _has_transition_phrase(self, normalized_text: str) -> bool:
        padded = f" {normalized_text} "
        return any(f" {phrase} " in padded for phrase in self.transition_phrases)

    def _has_overlap_phrase(self, normalized_text: str) -> bool:
        padded = f" {normalized_text} "
        return any(f" {phrase} " in padded for phrase in self.overlap_phrases)

    def _token_exists(self, token: str) -> bool:
        if not self.nlp:
            return False
        doc = self.nlp(token)
        if not doc:
            return False
        return not doc[0].is_oov

    def _get_close_match(self, token: str) -> Optional[str]:
        if not self.mapping_keys:
            return None
        matches = difflib.get_close_matches(token, self.mapping_keys, n=1, cutoff=self.validation_threshold)
        return matches[0] if matches else None

    def process_text(self, text: str) -> str:
        normalized_text = self._normalize(text)
        tokens = normalized_text.split()
        corrected_tokens = list(tokens)

        lemma_text = None
        if self.nlp:
            doc = self.nlp(text)
            lemma_text = " ".join(token.lemma_.lower() for token in doc if token.text.strip())

        for index, token in enumerate(tokens):
            if self._token_exists(token):
                continue
            corrected = self._get_close_match(token)
            if corrected and corrected != token:
                print(f"[PlaybackPipeline] Corrected '{token}' -> '{corrected}'")
                corrected_tokens[index] = corrected

        corrected_text = " ".join(corrected_tokens)
        corrected_lemma_text = None
        if self.nlp:
            doc = self.nlp(corrected_text)
            corrected_lemma_text = " ".join(token.lemma_.lower() for token in doc if token.text.strip())

        matches = self._find_hotword_matches(corrected_text, corrected_lemma_text)
        background_matches = [(hotkey, cfg, pos) for hotkey, cfg, pos in matches if cfg.get("category") == "background"]
        event_matches = [(hotkey, cfg, pos) for hotkey, cfg, pos in matches if cfg.get("category") == "event"]

        is_transition = self._has_transition_phrase(corrected_text)
        is_overlap = self._has_overlap_phrase(corrected_text)
        played: set[str] = set()

        background_to_play = []
        if background_matches:
            if is_overlap:
                background_to_play = background_matches
            else:
                background_to_play = [background_matches[-1]]

            if len(background_matches) > 1:
                print(f"[PlaybackPipeline] Background chain detected. Transition={is_transition}, overlap_phrase={is_overlap}")
                print(f"[PlaybackPipeline] Selected background hotword(s): {[hotkey for hotkey, _, _ in background_to_play]}")

        for bg_hotkey, bg_cfg, _ in background_to_play:
            print(f"[PlaybackPipeline] Triggered background hotword: '{bg_hotkey}'")
            self.playback.play_sound(file_path=bg_cfg["file"], channel_id=bg_cfg["channel"], is_loop=bg_cfg["is_loop"])
            played.add(bg_hotkey)

        for hotkey, cfg, _ in event_matches:
            if hotkey not in played:
                print(f"[PlaybackPipeline] Triggered event hotword: '{hotkey}'")
                self.playback.play_sound(file_path=cfg["file"], channel_id=cfg["channel"], is_loop=cfg["is_loop"])
                played.add(hotkey)

        return corrected_text if corrected_text != normalized_text else text

    def close(self):
        self.playback.stop_all()