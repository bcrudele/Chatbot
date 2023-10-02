from gtts import gTTS
import pygame.mixer

def tts(string):
    tts = gTTS(text=string, lang='en', slow=False)

    # File Save
    audio_filename = "google_tts.mp3"
    tts.save(audio_filename)

    # Initialize pygame.mixer
    pygame.mixer.init()
    pygame.mixer.music.load(audio_filename)
    pygame.mixer.music.play()

    # Wait for the audio to finish playing
    while pygame.mixer.music.get_busy():
        continue