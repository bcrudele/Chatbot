import google_tts   # Has Google TTS
import gpt_import   # Has ChatGPT 
import speech_recog # Has Speech Recognition

if __name__ == "__main__":
    command = speech_recog.user_listen()                    # Gets user input
    result = gpt_import.communicate_with_openai(command)    # Sends user input to ChatGPT
    google_tts.tts(result)                                  # Send ChatGPT result to TTS output
    print(result)                                           # Print to terminal