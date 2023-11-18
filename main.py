import google_tts   # Has Google TTS
import gpt_import   # Has ChatGPT 
import speech_recog # Has Speech Recognition
import time         # For runtime report
import weather_scrape as ws # For Weather
import news_fetch as nf     # For News
from googlesearch import search
import requests
from bs4 import BeautifulSoup

iter = 0            # Keeps track of iterations
program_stop_command_prompt = 'hippopotamus'
web_prompt = 'search'
weather_prompt = 'weather'
news_prompt = 'news'

if __name__ == "__main__":
    while True:
        command = speech_recog.user_listen() # Gets user input
        #command = 'the news'                # testing prompt
        print(command)
        
        if program_stop_command_prompt in command:
            print('Program stopped.')
            exit()

        elif weather_prompt in command:
            print("Searching weather data...")
            data, state, city = ws.weather_scrape(command)
            print(data)         # returns data in the form [temp, outlook, range, wind speed]
            print(state, city)
            google_tts.tts(data)

        elif news_prompt in command:
            news_category, news_data = nf.news_list()
            google_tts.tts(f'Heres the latest news on {news_category}: {news_data}')
            
        else:
            start = time.time()
            result = gpt_import.communicate_with_openai(command)    # Sends user input to ChatGPT
            end = time.time()
            run_time = end - start
            iter += 1
            print(f'Response {iter} - GPT Run-time: {run_time : .2f}s')
            print(result)
            google_tts.tts(result)                                  # Send ChatGPT result to TTS output
            
        
        '''Notes for Later:
            - Improve run-time through a cache
            - Clean main.py
            - Create a clean function to remove google_tts().mp3 files or convert them as temp files
            - Integrate personalities
            - Force references to old chat logs on initial request

            IDEAS: 
            
        # Personal Reminders
        def set_reminder(task, time):
            # Placeholder for reminder implementation
            pass

        # Random Facts
        def get_random_fact():
            # Placeholder for random facts implementation
            pass

        # Jokes and Humor
        def get_joke():
            # Placeholder for joke API integration or custom joke generation
            pass

        # Language Games
        def play_word_game():
            # Placeholder for word game implementation
            pass

        # Philosophical Conversations
        def engage_in_philosophy():
            # Placeholder for philosophical conversation prompts and responses
            pass

        # Music Recommendations
        def recommend_music(preferences):
            # Placeholder for music API integration and recommendation logic
            pass

        # Language Translation
        def translate_text(text, target_language):
            # Placeholder for translation implementation
            pass

        # Learning and Personalization
        def learn_from_user_preferences(preferences):
            # Placeholder for learning mechanism implementation
            pass

        # Mood Recognition
        def analyze_mood(user_input):
            # Placeholder for sentiment analysis implementation
            pass

        '''