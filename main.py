from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import base64
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import mediapipe as mp

app = FastAPI()

# Enable CORS (if testing from device)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
model = tf.keras.models.load_model("asl_model_mediapipe.h5")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)


def decode_base64_image(base64_string):
    decoded = base64.b64decode(base64_string)
    image = Image.open(BytesIO(decoded)).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def extract_landmarks(image):
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_hand_landmarks:
        return None
    hand_landmarks = results.multi_hand_landmarks[0]
    return [v for lm in hand_landmarks.landmark for v in (lm.x, lm.y, lm.z)]


def predict_letter(landmarks):
    input_data = np.array([landmarks])
    output = model.predict(input_data)
    return chr(ord('A') + np.argmax(output))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection accepted")
    
    frame_count = 0  # To count received frames for logging
    while True:
        try:
            data = await websocket.receive_text()
            frame_count += 1
            print(f"[{frame_count}] Received frame data (length={len(data)})")

            frame = decode_base64_image(data)
            print(f"[{frame_count}] Decoded image shape: {frame.shape}")

            landmarks = extract_landmarks(frame)
            if landmarks:
                print(f"[{frame_count}] Landmarks extracted: {len(landmarks)} values")
                letter = predict_letter(landmarks)
                print(f"[{frame_count}] Predicted letter: {letter}")
            else:
                print(f"[{frame_count}] No hand landmarks found")
                letter = "None"

            await websocket.send_text(letter)

        except Exception as e:
            print(f"[{frame_count}] Error: {e}")
            await websocket.send_text("Error")

