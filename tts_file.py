import pygame.mixer
import torch
from TTS.api import TTS
import numpy as np
from pydub import AudioSegment
from pydub.playback import play

def tts(string):
    freq_adj = 19600        # TBD adjustment

# Initialize the mixer (you only need to do this once)
    pygame.mixer.init(frequency=freq_adj)

# Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"

# List available TTS models and choose the first one, ft. https://github.com/coqui-ai/TTS
# list -> TTS().list_models()
    model_name = "tts_models/en/ljspeech/glow-tts"
# model_name = "tts_models/en/ljspeech/tacotron2-DDC"     # old model

# Initialize TTS
    tts = TTS(model_name).to(device)

# Use TTS
    text_to_convert = string
    audio_samples = tts.tts(text_to_convert, speaker=tts.speakers, language=tts.languages)

# Convert audio samples to integer format (16-bit signed PCM)
    audio_samples_int = (np.array(audio_samples) * 32767).astype(np.int16)    # '32767' maps the floats as 16 bits

# Convert audio samples to bytes
    audio_bytes = audio_samples_int.tobytes()

# Create a PyDub AudioSegment from the bytes
    audio_segment = AudioSegment(
        data=audio_bytes,
        sample_width=audio_samples_int.dtype.itemsize,
        frame_rate=freq_adj,  # Adjust the sample rate as needed
        channels=1,  # 1 channel for mono audio
    )

# PyDub effects
    audio_segment = audio_segment.speedup(playback_speed=1.5)   # Speed
#audio_segment = audio_segment.apply_gain(10)               # Gain

# Play the audio
    play(audio_segment)

# Buffer time (not needed)
# pygame.time.wait(1000)  

    pygame.mixer.quit()
