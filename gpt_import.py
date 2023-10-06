import config       # Has API & File Path
import openai       # ChatGPT API
 
def communicate_with_openai(prompt):
    openai.api_key = config.OPENAI_API_KEY 

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",   # Find best chat model later
        messages=[
            {"role": "system", "content": "You are my butler-like companion, speak to me as a friend."},
            {"role": "user", "content": prompt}
        ],
        max_tokens = 100         # Adjust for testing
    )

    return response['choices'][0]['message']['content']
"""
if __name__ == "__main__":
    prompt = "Tell me a funny story"       # Pull this from voice recognition later
    result = communicate_with_openai(prompt)
    #tts_file.tts(result)
    google_tts.tts(result)
    print(result)
"""