import pyaudio
import numpy as np
import pygame  # Do odtwarzania efektów
from faster_whisper import WhisperModel
from collections import deque

# 1. Inicjalizacja Audio (Pygame)
pygame.mixer.init()

# Twoja baza dźwięków: słowo_kluczowe -> plik_audio
SOUND_DB = {
    "las": "sounds/forest_ambient.mp3",
    "miecz": "sounds/sword_clash.wav",
    "ogień": "sounds/fire_crackle.mp3",
    "woda": "sounds/water_splash.wav"
}

def play_effect(word):
    if word in SOUND_DB:
        print(f"\n>>> AKCJA: Wykryto '{word}', odtwarzam: {SOUND_DB[word]}")
        pygame.mixer.Sound(SOUND_DB[word]).play()

# 2. Konfiguracja Modelu (Zoptymalizowana pod Twoje CPU)
model = WhisperModel("base", device="cpu", compute_type="int8")

# 3. Parametry Nagrywania
RATE = 16000
CHUNK = 4000 
audio_buffer = deque(maxlen=10) # ~2.5 sekundy kontekstu

audio_interface = pyaudio.PyAudio()
stream = audio_interface.open(format=pyaudio.paInt16, channels=1, rate=RATE, 
                              input=True, frames_per_buffer=CHUNK)

print("System gotowy. Opowiadaj historię...")

processed_text = "" # Zapobiega wielokrotnemu wyzwalaniu tego samego dźwięku

try:
    while True:
        data = stream.read(CHUNK)
        audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        audio_buffer.append(audio_chunk)

        full_audio = np.concatenate(list(audio_buffer))
        
        # Transkrypcja z małym beam_size dla szybkości
        segments, _ = model.transcribe(full_audio, language="pl", beam_size=1)

        for segment in segments:
            current_text = segment.text.lower()
            
            # Sprawdzamy każde słowo z bazy w nowym tekście
            for keyword in SOUND_DB.keys():
                # Sprawdź czy słowo jest w tekście I czy nie wyzwoliliśmy go przed chwilą
                if keyword in current_text and keyword not in processed_text:
                    play_effect(keyword)
            
            processed_text = current_text # Aktualizujemy kontekst "przeczytany"
            print(f"\rLektor: {segment.text}", end="")

except KeyboardInterrupt:
    print("\nWyłączanie...")
finally:
    stream.stop_stream()
    stream.close()
    audio_interface.terminate()