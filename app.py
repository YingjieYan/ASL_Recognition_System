from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import tensorflow as tf
import time
import mediapipe as mp
from collections import deque, Counter

app = Flask(__name__)

# Sign language class labels
class_names = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'
]

# Load trained model
model = tf.keras.models.load_model("trained_asl_landmark_mlp_local.keras")

# Initialize webcam
camera = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Recognition state
text_buffer = ""
current_prediction = ""
last_time = time.time()
prediction_window = deque(maxlen=3)

# Normalize hand landmarks
def normalize_landmarks(landmarks_world_coords):
    if landmarks_world_coords is None or len(landmarks_world_coords) == 0:
        return None
    wrist = landmarks_world_coords[0].copy()
    normalized_coords = landmarks_world_coords - wrist
    palm_size = np.linalg.norm(normalized_coords[9])
    if palm_size < 1e-6:
        return None
    normalized_coords /= palm_size
    return normalized_coords.flatten()

# Generate video stream with recognition and hand landmark drawing
def generate_frames():
    global text_buffer, current_prediction, last_time, prediction_window

    while True:
        success, frame = camera.read()
        if not success:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]

            # Draw hand landmarks and connections on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Prepare input for model
            landmarks_np = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            normalized = normalize_landmarks(landmarks_np)

            current_time = time.time()
            if current_time - last_time >= 0.4:  # recognition interval
                if normalized is not None and len(normalized) == model.input_shape[-1]:
                    input_array = np.expand_dims(normalized, axis=0)
                    prediction = model.predict(input_array)[0]

                    index = int(np.argmax(prediction))
                    confidence = float(np.max(prediction))

                    if 0 <= index < len(class_names):
                        label = class_names[index]
                        print(f"[Prediction] {label} (Confidence: {confidence:.2f})")

                        if confidence > 0.8:
                            prediction_window.append(label)
                            if len(prediction_window) == 3: 
                                most_common, count = Counter(prediction_window).most_common(1)[0]
                                if count == 3 and most_common != "nothing":
                                    if most_common == "space":
                                        text_buffer += " "
                                    elif most_common == "del":
                                        text_buffer = text_buffer[:-1]
                                    else:
                                        text_buffer += most_common
                                    current_prediction = text_buffer
                                    prediction_window.clear()
                            last_time = current_time

        # Draw text buffer on frame
        cv2.putText(frame, current_prediction, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        # Convert frame to JPEG for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_text')
def get_text():
    return jsonify({'text': current_prediction})

@app.route('/clear_text')
def clear_text():
    global text_buffer, current_prediction, prediction_window
    text_buffer = ""
    current_prediction = ""
    prediction_window.clear()
    return jsonify({'status': 'cleared'})

if __name__ == '__main__':
    app.run(debug=True)
