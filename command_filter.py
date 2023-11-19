import google_tts   # Has Google TTS
import gpt_import   # Has ChatGPT 
import time         # For runtime report
import weather_scrape as ws # For Weather
import news_fetch as nf     # For News
import main

#web_prompt = 'search'
weather_prompt = 'weather'
news_prompt = 'news'

def text_decomp(command):

    if main.program_stop_command_prompt in command:
        return False, False 

    elif weather_prompt in command:
        print("Searching weather data...")
        data, state, city = ws.weather_scrape(command)
        print(data)         # returns data in the form [temp, outlook, range, wind speed]
        print(state, city)
        google_tts.tts(data)
        return True, True

    elif news_prompt in command:
        news_category, news_data = nf.news_list()
        google_tts.tts(f'Heres the latest news on {news_category}: {news_data}')
        return True, True
        
    else:
        start = time.time()
        result = gpt_import.communicate_with_openai(command)    # Sends user input to ChatGPT
        end = time.time()
        run_time = end - start
        print(f'- GPT Run-time: {run_time : .2f}s')
        print(result)
        google_tts.tts(result)                                  # Send ChatGPT result to TTS output
        return True, True