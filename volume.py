import cv2
import mediapipe as mp
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize volume control
def init_volume_control():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    return cast(interface, POINTER(IAudioEndpointVolume))

# Get the current volume
def get_volume(audio_volume):
    return audio_volume.GetMasterVolumeLevelScalar()

# Set the volume
def set_volume(audio_volume, level):
    audio_volume.SetMasterVolumeLevelScalar(level, None)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Create OpenCV window
cv2.namedWindow("Hand Tracking")

# Initialize OpenCV camera capture
cap = cv2.VideoCapture(0)

# Initialize volume control
volume = init_volume_control()

# Initialize MediaPipe Hands
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to get hand landmarks
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                # Extract the landmarks for the thumb and index finger
                thumb_tip = landmarks.landmark[4]
                index_finger_tip = landmarks.landmark[8]

                # Calculate the Euclidean distance between thumb and index finger tips
                finger_distance = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_finger_tip.x, index_finger_tip.y]))

                # Map finger distance to volume level
                max_distance = 0.2  # Adjust as needed
                min_distance = 0.0   # Adjust as needed
                volume_level = 1.0 - (finger_distance - min_distance) / (max_distance - min_distance)
                volume_level = max(0.0, min(1.0, volume_level))

                # Set the volume based on finger proximity
                set_volume(volume, volume_level)

                # Show the finger proximity and volume level on the screen
                cv2.putText(frame, f"Finger Proximity: {finger_distance:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Volume: {volume_level:.2f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Hand Tracking", frame)

        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
