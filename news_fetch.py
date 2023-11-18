from newsapi import NewsApiClient
import config
import google_tts
import speech_recog as sr

def check_word_presence(input_string, target_word):
    return target_word.lower() in input_string.lower()

def news_list():
    newsapi = NewsApiClient(api_key=config.NEWS_API_KEY)

    # input_country = 'United States'
    input_country_code = 'us'   # in alpha2 format, use pycountry if needed

    valid_search_categories = ['business','entertainment','general','health','science','technology']
    option = 'NULL'

    google_tts.tts('What category of news would you like?')    # Make this randomly generated

    while option not in valid_search_categories:
        option = sr.user_listen()
    
        for category in valid_search_categories:
            if check_word_presence(option.lower(), category):
                option = category
        
        if option not in valid_search_categories:
            print('Please try another category')
            google_tts.tts('Please try another category')

    print(f'Selected category: {option}')

    top_headlines = newsapi.get_top_headlines(

    category=f'{option.lower()}', language='en', country=f'{input_country_code}')

    Headlines = top_headlines['articles']
    data = []                       # holds all articles in output

    if Headlines:
        for articles in Headlines:
            # Using find instead of index to avoid ValueError if '-' is not found
            b = articles['title'][::-1].find("-")
            
            # Check if '-' is found before using it to slice the string
            if b != -1:
                data.append(articles['title'][:-b-2])
            else:
                data.append(articles['title'])
    else:
        print(f"Error in News_fetch")
        data[0] = 'Error'

    return option, data 