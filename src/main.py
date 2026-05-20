from src.pipelines.transcription import TranscriptionPipeline

def on_transcription(text: str):
    print(f"Transcribed text: {text}")

if __name__ == "__main__":
    transcriptor = TranscriptionPipeline(on_transcription)
    transcriptor.run()