import os
import requests
import json
from pathlib import Path

SAVE_DIR: str = "sounds"
API_KEY: str = "" 

class DownloadSounds:
    def __init__(self, sound_mapping: dict = None):
        self.sound_mapping = sound_mapping or {}
        
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)

    def download_all(self):
        for name, config in self.sound_mapping.items():
            file_key = config.get('file_key')
            if not file_key:
                continue

            if os.path.exists(os.path.join(SAVE_DIR, f"{file_key}.mp3")):
                print(f"File already exists for {file_key}, skipping download.")
                continue

            url = f"https://freesound.org/apiv2/search/text/?query={file_key}&token={API_KEY}&fields=id,name,previews"
        
            try:
                response = requests.get(url)
                response.raise_for_status()
                data = response.json()

                if not data.get("results"):
                    print(f"No results found on Freesound for: {file_key}")
                    continue

                first = data["results"][0]
                
                download_url = first["previews"]["preview-hq-mp3"]

                print(f"Downloading {file_key}...")
                
                sound_data = requests.get(download_url)
                sound_data.raise_for_status()

                file_path = os.path.join(SAVE_DIR, f"{file_key}.mp3")
                with open(file_path, "wb") as f:
                    f.write(sound_data.content)
                    
                print(f"Success! Saved to {file_path}")

            except Exception as e:
                print(f"Error downloading {file_key}: {e}")

if __name__ == "__main__":
    config_path = Path(__file__).parent.parent / "utils" / "playback_sound_mapping.json"
    
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
            
        downloader = DownloadSounds(config)
        downloader.download_all()
        
    except FileNotFoundError:
        print(f"Could not find the JSON file at: {config_path}")
    except json.JSONDecodeError:
        print("The JSON file is poorly formatted.")