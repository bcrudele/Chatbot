import google_tts   # Has Google TTS
import speech_recog # Has Speech Recognition

if __name__ == "__main__":
    command = speech_recog.user_listen()
    # prompt = "Tell me a funny story"       # Pull this from voice recognition later
    # result = communicate_with_openai(prompt)
    google_tts.tts(command)                   # Change to 'result' later
    print(command)                            # Change to 'result' later