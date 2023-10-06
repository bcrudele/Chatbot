import speech_recognition as sr
# https://pypi.org/project/SpeechRecognition/

delay = 2  # timeout var in listen()

def user_listen():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Start Speaking...")
        recognizer.adjust_for_ambient_noise(source)

        # timeout: the maximum time to wait for audio before timing out
        # phrase_time_limit: the maximum time allowed for speaking a phrase
        audio = recognizer.listen(source, timeout=delay)

    try:
        return recognizer.recognize_google(audio)
    except sr.RequestError:
        print("Unable to access the Google API")
        return None