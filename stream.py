import streamlit as st
import cv2
import time
from datetime import datetime
import numpy as np
from collections import deque
from keras.models import load_model
import telepot
import os

bot = telepot.Bot('8197726841:AAHnxQNnP_6EVkhvSxk_45N4T30RDfPmdJA')  

# Ensure the upload folder exists
UPLOAD_FOLDER = 'uploaded_videos'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def save_annotated_video(input_video, output_video, telegram_group_id):
    st.write("Loading model...")
    model = load_model('modelnew.h5')
    Q = deque(maxlen=128)


    if isinstance(input_video, int):
        vs = cv2.VideoCapture(input_video)
    else:
        vs = cv2.VideoCapture(input_video)

    (W, H) = (None, None)
    violence_detected = False
    violence_start_frame = None
    frame_count = 0

    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video, fourcc, 30.0, (1280, 720))  

    smoothing_window = 3
    prediction_history = deque(maxlen=smoothing_window)

    stframe = st.empty()

    while True:
        (grabbed, frame) = vs.read()

        if not grabbed:
            break

        if W is None or H is None:
            (H, W) = frame.shape[:2]

        output = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (128, 128)).astype("float32")
        frame = frame.reshape(128, 128, 3) / 255

        preds = model.predict(np.expand_dims(frame, axis=0))[0]
        Q.append(preds)

        results = np.array(Q).mean(axis=0)
        i = (preds > 0.50)[0]
        prediction_history.append(i)

        smoothed_prediction = np.mean(prediction_history) > 0.5
        label = smoothed_prediction

        text_color = (0, 255, 0)

        if label:
            text_color = (0, 0, 255)

            if not violence_detected:
                violence_detected = True
                violence_start_frame = frame_count
                violence_start_time = time.time()
        else:
            violence_detected = False

        if violence_detected and frame_count == violence_start_frame + 10:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            message = f"Violence detected at {current_time}"
            with open('alert_frame.jpg', 'wb') as f:
                cv2.imwrite('alert_frame.jpg', frame * 255)
                bot.sendPhoto(telegram_group_id, open('alert_frame.jpg', 'rb'), caption=message)

        text = "Violence: {}".format(label)
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(output, text, (35, 50), FONT, 1.25, text_color, 3)

        
        out.write(output)

        stframe.image(output, channels="BGR")

        frame_count += 1

    st.write("Processing complete. Cleaning up...")
    vs.release()
    out.release()

st.title("Real-Time Violence Detection")

source_option = st.radio(
    "Choose Input Source",
    options=["Webcam", "Upload a Video"]
)

telegram_group_id = st.text_input("Telegram Group ID", "1188696687")

if source_option == "Webcam":
    if st.button("Start Detection"):
        save_annotated_video(0, "annotated_webcam.avi", telegram_group_id)
elif source_option == "Upload a Video":
    video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    if video_file is not None:
        video_path = os.path.join(UPLOAD_FOLDER, video_file.name)
        with open(video_path, "wb") as f:
            f.write(video_file.read())
        if st.button("Start Detection"):
            save_annotated_video(video_path, "annotated_video.avi", telegram_group_id)
