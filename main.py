import google_tts   # Has Google TTS
import gpt_import   # Has ChatGPT 
import speech_recog # Has Speech Recognition
import time         # For runtime
import weather_scrape as ws # For Weather
from googlesearch import search
import requests
from bs4 import BeautifulSoup

iter = 0            # Keeps track of iterations
program_stop_command_prompt = 'hippopotamus'
web_prompt = 'search'
weather_prompt = 'weather'

if __name__ == "__main__":
    while True:
        command = speech_recog.user_listen() # Gets user input
        print(command)
        if program_stop_command_prompt in command:
            print('Program stopped.')
            break
        if weather_prompt in command:
            print("Searching weather data...")
            city = 'West-Lafayette'    # add UI later
            state = 'IN'               # add UI later
            data = ws.weather_scrape(city, state)
            print(data)         # returns data in the form [temp, outlook, range, wind speed]
        else:
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