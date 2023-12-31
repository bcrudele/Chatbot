import speech_recognition as sr
# https://pypi.org/project/SpeechRecognition/

delay = 1.5  # timeout var in listen()
def user_listen():
    
    recognizer = sr.Recognizer()
    while True:
        with sr.Microphone() as source:
            print("Start Speaking...")

            recognizer.adjust_for_ambient_noise(source)

            # timeout: the maximum time to wait for audio before timing out
            # phrase_time_limit: the maximum time allowed for speaking a phrase
            try:
                audio = recognizer.listen(source, timeout=delay)
            except sr.WaitTimeoutError:
                print("Try again, couldn't hear you:\n")
                continue
            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand the audio.")
                continue
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")
                continue
        

        try:
            print(recognizer.recognize_google(audio))
            return recognizer.recognize_google(audio)
        except sr.RequestError:
            print("Unable to access the Google API")
            return None
