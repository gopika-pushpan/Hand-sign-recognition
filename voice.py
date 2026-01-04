def get_voice_letter():
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("🎙️ Adjusting for ambient noise...")
            r.adjust_for_ambient_noise(source, duration=1)

            print("🎙️ Listening for a single letter (like A-Z)...")
            audio = r.listen(source, timeout=5, phrase_time_limit=3)

        text = r.recognize_google(audio)
        print("🗣️ You said:", text)
        
        if len(text) == 1 and text.isalpha():
            return text.upper()
        else:
            print("❗ Say just a single letter clearly (A to Z).")
            return ""
    except sr.WaitTimeoutError:
        print("⌛ Timed out: No voice detected.")
        return ""
    except sr.UnknownValueError:
        print("❌ Didn't understand. Try again.")
        return ""
    except sr.RequestError as e:
        print(f"🚫 Could not request results; check internet. Error: {e}")
        return ""
