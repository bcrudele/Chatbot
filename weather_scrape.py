import requests
from bs4 import BeautifulSoup
from googlesearch import search

def weather_scrape(city, state):

    query = f"weather.com today today {city} {state}"
    results = []
    for result in search(query):
        results.append(result)
        #print(result)

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

        # Outlook
        outlook = soup.find('div', class_='CurrentConditions--phraseValue--mZC_p')
        if outlook:
            outlook_text = outlook.get_text().strip()
            #print(f"Outlook: {outlook_text}")

        # Range
        range = soup.find('div', class_='CurrentConditions--tempHiLoValue--3T1DG')
        if range:
            range_text = range.get_text().strip()
            for word in stripper:
                range_text = range_text.replace(word, '')
            #print(f"Range: {range_text}")

        # Wind speed
        wind = soup.find('span', class_='Wind--windWrapper--3Ly7c')
        if wind:
            wind_text = wind.get_text().strip()
    
            for word in stripper:
                wind_text = wind_text.replace(word, '')
            #print(f"Wind: {wind_text}")

        data = f'{temperature_text}' + f'{outlook_text}' + f'{range_text}' + f'{wind_text}'

    return data