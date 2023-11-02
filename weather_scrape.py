import requests
from bs4 import BeautifulSoup
from googlesearch import search
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk import word_tokenize, pos_tag
import time
from requests.exceptions import HTTPError

def location(command):
    tokens = word_tokenize(command[0].lower() + command[1:]) # lowercases first word
    tags = pos_tag(tokens)
    city = ''    
    state = ''   

    for word, tag in tags:
        if tag == 'NNP':
            if not city:
                city = word
            else:
                state = word

    return(state, city)
 

def weather_scrape(command):

    state, city = location(command)  # State and City finder

    #query = f"weather.com today today {city} {state}"
    query = f"stock broker today"
    results = []
    try:
        for result in search(query):
            #time.sleep(2)
            results.append(result)
            print(result)
            if len(results) >= 5:
                break

    except HTTPError as error:
        if error.response.status_code == 429:
            print("Rate limit exceeded")
            return(None, state, city)

    url = f'{results[0]}'
    # Send an HTTP GET request to the URL
    response = requests.get(url)
    stripper = ['Wind Direction', 'Day', 'Night', '•']
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        # Temp
        temperature = soup.find('span', class_='CurrentConditions--tempValue--MHmYY')
        if temperature:
            temperature_text = temperature.get_text().strip()
            #print(f"Temperature: {temperature_text}")
        else:
            temperature_text = ''

        # Outlook
        outlook = soup.find('div', class_='CurrentConditions--phraseValue--mZC_p')
        if outlook:
            outlook_text = outlook.get_text().strip().lower()
            #print(f"Outlook: {outlook_text}")
        else:
            outlook_text = ''

        # Range
        range = soup.find('div', class_='CurrentConditions--tempHiLoValue--3T1DG')
        if range:
            range_text = range.get_text().strip()
            for word in stripper:
                range_text = range_text.replace(word, '')
            #print(f"Range: {range_text}")
        else:
            range_text = ''

        # Wind speed
        wind = soup.find('span', class_='Wind--windWrapper--3Ly7c')
        if wind:
            wind_text = wind.get_text().strip()
    
            for word in stripper:
                wind_text = wind_text.replace(word, '')
            #print(f"Wind: {wind_text}")
        else:
            wind_text = ''
        #data = f'{temperature_text}' + f'{outlook_text}' + f'{range_text}' + f'{wind_text}'
        
        if outlook_text == 'clear' or 'partly cloudy':
            outlook_text = f'{outlook_text} skies'
        elif outlook_text == 'rainy':
            outlook_text = 'rain on the forecast'

        data = f'{city} {state} is currently {temperature_text} degrees with {outlook_text}'
        print(data)
    return(data, state, city)

#weather_scrape("Tell me the weather in Miami, Florida!")