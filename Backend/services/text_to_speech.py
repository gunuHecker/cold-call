import pyttsx3

def text_to_speech(text):
    engine = pyttsx3.init()
    filename = f"audio_response.mp3"
    engine.save_to_file(text, filename)
    engine.runAndWait()
    return filename
