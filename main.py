import google_tts   # Has Google TTS
import gpt_import   # Has ChatGPT 
import speech_recog # Has Speech Recognition
import time         # For runtime

iter = 0            # Keeps track of iterations

if __name__ == "__main__":
    while True:
        command = speech_recog.user_listen()                    # Gets user input
        start = time.time()
        result = gpt_import.communicate_with_openai(command)    # Sends user input to ChatGPT
        end = time.time()
        run_time = end - start
        iter += 1
        print(f'Response {iter} - GPT Run-time: {run_time : .2f}s')
        google_tts.tts(result, iter)                                  # Send ChatGPT result to TTS output
        print(result)
        
        '''Notes for Later:
            - Improve run-time through a cache
            - Clean main.py
            - Create a clean function to remove google_tts().mp3 files or convert them as temp files
            - Integrate personalities
            - Force references to old chat logs on initial request
        '''