import speech_recognition as sr
# https://pypi.org/project/SpeechRecognition/

def listen_for_voice_command():
    recognizer = sr.Recognizer()
    bot_name = "felix"

    while True:
        with sr.Microphone() as source:
            print("Listening for a command...")
            try:
                audio = recognizer.listen(source, timeout=5)  # Adjust timeout as needed
                command = recognizer.recognize_google(audio).lower()

                if bot_name in command:
                    print(f"Command received: {command}")

            except sr.UnknownValueError:
                print("Could not understand audio")
            except sr.RequestError as e:
                print(f"Error with the speech recognition service: {e}")
