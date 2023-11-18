from gtts import gTTS      # Google TTS
import pygame.mixer        # Audio Output Library
from datetime import datetime
import os

def tts(string):
    tts = gTTS(text=string, lang='en', slow=False)

    # File Save
    base_name = "google_tts"
    ext = ".mp3"
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    audio_filename = f"{base_name}_{timestamp}{ext}"

    tts.save(audio_filename)

    # Initialize pygame.mixer
    pygame.mixer.init()
    pygame.mixer.music.load(audio_filename)
    pygame.mixer.music.play()

    # Wait for the audio to finish playing
    while pygame.mixer.music.get_busy():
        continue

def cleanup_audio(audio_filename):
    if os.path.exists(audio_filename):
        os.remove(audio_filename)