# Hand-sign-recognition
An end-to-end computer vision and deep learning project that recognizes hand sign alphabets (A–Z) in real time using a webcam and converts them into text and speech, enabling accessible human–computer interaction.

🔍 Project Objective

The objective of this project is to design and implement a real-time hand sign recognition system that can:

Detect hand gestures using a webcam

Recognize sign language alphabets (A–Z)

Convert continuous gestures into readable words

Generate speech output from recognized text

This project demonstrates a complete AI pipeline — from data collection and preprocessing to model training and real-time deployment.

🧩 Dataset Description

The dataset is custom-built using a webcam and contains hand gesture images representing alphabets A to Z.

Dataset characteristics:

Alphabet-based gesture images (A–Z)

Images captured under different lighting and angles

Class-wise organization for each alphabet

Dataset collected manually to simulate real-world conditions

⚠️ The dataset is not included in the repository due to size limitations and can be recreated using the data collection script.

🛠️ Tools & Technologies Used

Python — core programming language

OpenCV — real-time video capture and image processing

MediaPipe — hand landmark detection and tracking

TensorFlow / Keras — CNN model development and training

NumPy — numerical computation

SpeechRecognition — speech-to-text input

pyttsx3 — text-to-speech conversion

GitHub — version control and project hosting

🧪 Data Collection & Preparation

Hand gesture images collected using webcam

Images organized class-wise (A–Z)

Dataset validated for consistency and labeling accuracy

Preprocessed images resized and normalized before training

Separate scripts used for data collection and preprocessing

🧠 Model Development

Model Type: Convolutional Neural Network (CNN)

Input: Hand images / hand landmarks

Output: Alphabet class (A–Z)

Model trained on the custom dataset

Achieved high training accuracy

The trained model is then integrated into a real-time prediction pipeline.

🎥 Real-Time Gesture Recognition

The real-time system performs the following steps:

Captures video frames using webcam

Detects hand landmarks using MediaPipe

Extracts features from detected hand

Predicts the corresponding alphabet

Displays prediction on screen

Forms words from continuous predictions

Converts final text into speech

🎮 Application Controls
Action	Input
Show hand gesture	Detect alphabet
V key	Speak a letter using voice input
S key	Speak the complete word
ESC key	Exit the application
📊 Results & Observations

The system successfully recognizes most alphabet gestures in real time

Prediction accuracy varies depending on:

Lighting conditions

Hand orientation

Similarity between gestures

Some letters with similar hand shapes may cause minor misclassification

Despite these challenges, the system performs reliably under controlled conditions.

⚠️ Limitations

Similar-looking gestures (e.g., M, N, S, T) are harder to distinguish

Performance depends on camera quality and lighting

Dataset collected from limited users affects generalization

Real-time predictions may fluctuate frame-to-frame

These limitations are common in real-time sign language recognition systems.

🌱 Future Enhancements

Support for numbers (0–9)

Sentence-level sign recognition

Dictionary-based word correction

Multi-user dataset for improved accuracy

Web or mobile application deployment

📂 Repository Structure
hand-sign-recognition/
│
├── src/
│   ├── train_model.py        # Model training script
│   ├── predict_sign.py       # Real-time prediction script
│   └── voice.py              # Speech input/output
│
├── README.md
├── requirements.txt
└── .gitignore

🎓 Learning Outcomes

Practical experience with Computer Vision

Understanding real-time ML system constraints

End-to-end deep learning pipeline implementation

Integration of vision, ML, and speech technologies

Debugging and optimizing real-world AI applications

📜 License

This project is licensed under the MIT License.

✅ Final Note

This project showcases a complete real-time AI application, highlighting both the capabilities and challenges of hand sign recognition systems and demonstrating strong problem-solving, implementation, and debugging skills.
## ✍️ Author  

**Gopika Pushpan**  
B.Tech Computer Science | AI & Machine Learning Enthusiast
