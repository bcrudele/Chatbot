import threading
import time
import speech_recog

if __name__ == "__main__":

    # Sets thread (core) to constantly listen for input
    input_thread = threading.Thread(target=speech_recog.listen_for_voice_command)

    try:
        input_thread.start()

        # Common main
        while(True):
            print("5 seconds parsed\n")
            time.sleep(5)

    except KeyboardInterrupt:
        # Fix this line
        pass
    finally:
        # Wait for the input thread to finish before exiting
        input_thread.join()

    print("Main thread exiting.")