import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.symbol_database')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hide TensorFlow logs

import cv2
import mediapipe as mp
import joblib
import numpy as np
import pyttsx3
import speech_recognition as sr

# ===== Load trained model =====
model = joblib.load("model.pkl")

# ===== Initialize Mediapipe =====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# ===== Initialize text-to-speech =====
engine = pyttsx3.init()

# ===== Initialize speech recognizer =====
recognizer = sr.Recognizer()

# ===== Open webcam =====
cap = cv2.VideoCapture(0)

sentence = ""      # stores predicted letters/words
last_letter = None # to prevent spam when holding same sign

print("\n[📡] Prediction running...")
print("👋 Show hand gestures → letters appear")
print("🔤 Letter forms → word builds up")
print("🗣️ Press V → say a letter to add")
print("🔊 Press S → hear the full word")
print("🚪 Press ESC → quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y])

            # Match model input shape
            if len(landmarks) == model.n_features_in_:
                prediction = model.predict([landmarks])[0]

                # Only add if different from last predicted
                if prediction != last_letter:
                    sentence += prediction
                    last_letter = prediction

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        last_letter = None  # reset if no hand detected

    # ===== Display the sentence =====
    cv2.putText(frame, sentence, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # ===== Show instructions on screen =====
    cv2.putText(frame, "V: Voice input | S: Speak word | ESC: Quit",
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1)

    cv2.imshow("Sign Language Prediction", frame)

    key = cv2.waitKey(1) & 0xFF

    # Voice input
    if key == ord('v'):
        print("🎙️ Say a letter...")
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
            try:
                voice_text = recognizer.recognize_google(audio).strip().upper()
                if len(voice_text) == 1 and voice_text.isalpha():
                    sentence += voice_text
                    print(f"✅ Added letter: {voice_text}")
                else:
                    print("❌ Please say a single letter (A-Z).")
            except sr.UnknownValueError:
                print("❌ Could not understand voice")
            except sr.RequestError:
                print("❌ Speech recognition service unavailable")

    # Speak full word
    elif key == ord('s'):
        if sentence:
            print(f"🔊 Speaking: {sentence}")
            engine.say(sentence)
            engine.runAndWait()

    # Quit
    elif key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
