import cv2
import mediapipe as mp

# Initialize MediaPipe Hands and Drawing
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Set up drawing styles (red dots for landmarks, green lines for connections)
landmark_style = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4)
connection_style = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)

# Initialize the Hands model
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Start video capture
cap = cv2.VideoCapture(0)
print("📷 Starting camera... Press 'q' to exit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("⚠️ Failed to capture frame")
        continue

    # Flip and convert image to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe
    results = hands.process(rgb_frame)

    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_style,
                connection_style
            )

    # Show output
    cv2.imshow("🖐️ Hand Detection", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
