from gtts import gTTS      # Google TTS
import pygame.mixer        # Audio Output Library

def tts(string, iter):
    tts = gTTS(text=string, lang='en', slow=False)

    # File Save
    base_name = "google_tts"
    ext = ".mp3"
    audio_filename = f"{base_name}({iter}){ext}"  # has an over-write issue, fix later
    

    tts.save(audio_filename)

    # Initialize pygame.mixer
    pygame.mixer.init()
    pygame.mixer.music.load(audio_filename)
    pygame.mixer.music.play()

    # Wait for the audio to finish playing
    while pygame.mixer.music.get_busy():
        continue