import config       # Has API & File Path
import tts_file     # Has Coqui TTS
import google_tts   # Has Google TTS
import openai      
def communicate_with_openai(prompt):
    openai.api_key = config.OPENAI_API_KEY 

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",   # Find best chat model later
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens = 10         # Adjust for testing
    )

    return response['choices'][0]['message']['content']

if __name__ == "__main__":
    prompt = "Insert Prompt Here"       # Pull this from voice recognition later
    result = communicate_with_openai(prompt)
    # tts_file.tts(result)
    google_tts.tts(result)
    print(result)