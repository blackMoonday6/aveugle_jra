import os
import time
import playsound
import speech_recognition as sr
from gtts import gTTS


 def speak(text):
    tts = gTTS(text=text, lang="en")
    filename = "voice.mp3"
    tts.save(filename)
    playsound.playsound(filename)


def get_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
        said = ""

        try:
            said = r.recognize_google(audio)
            print(said)
        except Exception as e:
            print("Error: " + str(e))

    return said

text = get_audio()

if "hello" in text:
    speak("hello, how are you?")
elif "your name" in text:
    speak("My name is astro")
elif "go" in text:
    speak("we are going")
elif "stop" in text:
    speak("Ok sir")
elif "velocity up" in text:
    speak("My name is astro")
elif "velocity down" in text:
    speak("My name is astro")