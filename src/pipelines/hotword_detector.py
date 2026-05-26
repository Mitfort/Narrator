import difflib
import string
import unicodedata
from typing import Any


class HotwordDetector:
    def __init__(self, config: dict[str, Any]):
        self.validation_threshold = config.get("Validation_Threshold", 0.75)
        self.raw_mapping = config.get("Playback_Sound_Mapping", {})
        self.mapping_keys = []
        self._load_mapping_keys()

    def _normalize(self, s: str) -> str:
        normalized = unicodedata.normalize("NFKD", s)
        ascii_only = "".join(ch for ch in normalized if not unicodedata.combining(ch))
        translator = str.maketrans("", "", string.punctuation)
        return ascii_only.translate(translator).lower().strip()

    def _load_mapping_keys(self):
        for hotword in self.raw_mapping.keys():
            normalized = self._normalize(hotword)
            if normalized:
                self.mapping_keys.append(normalized)
        self.mapping_keys.sort(key=lambda key: len(key.split()), reverse=True)

    def _matches_fuzzy_phrase(self, tokens: list[str], hotword: str) -> bool:
        hot_tokens = hotword.split()
        if not hot_tokens:
            return False

        if len(hot_tokens) == 1:
            return self._matches_fuzzy_token(tokens, hotword)

        best_ratio = 0.0
        for start in range(len(tokens) - len(hot_tokens) + 1):
            window = " ".join(tokens[start:start + len(hot_tokens)])
            ratio = difflib.SequenceMatcher(None, window, hotword).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                if best_ratio >= self.validation_threshold:
                    return True

        return best_ratio >= self.validation_threshold

    def _matches_fuzzy_token(self, tokens: list[str], hotword: str) -> bool:
        if not tokens:
            return False
        matches = difflib.get_close_matches(hotword, tokens, n=1, cutoff=self.validation_threshold)
        return bool(matches)

    def has_hotword(self, text: str) -> bool:
        if not text:
            return False

        normalized_text = self._normalize(text)
        padded_text = f" {normalized_text} " if normalized_text else ""

        for hotword in self.mapping_keys:
            if f" {hotword} " in padded_text:
                return True

        tokens = normalized_text.split()
        if not tokens:
            return False

        for hotword in self.mapping_keys:
            if self._matches_fuzzy_phrase(tokens, hotword):
                return True

        return False
